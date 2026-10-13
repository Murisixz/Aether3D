"""The procedural system: parameters, geometry nodes, dependencies, evaluation and caching.

Model
-----
* A scene owns one :class:`ProceduralSystem` (``scene.procedural``).
* **Scene parameters** are typed values (or expressions over other scene
  parameters) shared by every procedural definition.
* A **geometry node** (:class:`ProceduralGeometry`) has its own parameters, a
  *source* -- a parametric primitive, a stored base mesh, or the output of
  another geometry node (optionally with parameter overrides, optionally taken
  after a named modifier) -- and an ordered stack of :class:`Modifier` s.
* Scene objects reference geometry nodes; several objects may share one node
  (and therefore one evaluated :class:`~aether3d.Mesh`).

Evaluation is pure and content-addressed: the cache key of each step is a
SHA-256 over the fully resolved inputs (primitive parameters, base-mesh hash,
upstream keys, modifier type and resolved parameters, operand keys). A change
anywhere produces new keys downstream, so results are never stale, while
unchanged prefixes of a stack are reused.

Every mutation is a transaction: the change is applied, references, schemas,
dependency cycles and the evaluation of every affected node are checked, and
on any failure the previous definitions are restored exactly. Successful
transactions form an undo/redo history.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import Diagnostic, worst_severity
from ..mesh import Mesh
from .modifiers import MODIFIER_TYPES, ModifierContext
from .params import (
    PARAMETER_TYPES,
    Expr,
    Parameter,
    ProceduralError,
    Rule,
    check_rules,
    check_type,
    evaluate_expression,
    make_parameter,
    make_rule,
    validate_values,
    value_from_json,
    value_to_json,
)
from .polymesh import PolyMesh, PolyMeshError, mesh_hash
from .primitives import PRIMITIVES

if TYPE_CHECKING:
    from ..scene import Scene


def _new_id() -> str:
    return str(uuid.uuid4())


def _canonical(value: Any) -> Any:
    if isinstance(value, Expr):
        return {"expr": value.source}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, float):
        return repr(value)  # exact
    return value


def _key(*parts: Any) -> str:
    text = json.dumps(_canonical(list(parts)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #


@dataclass
class Source:
    """Where a geometry node's base geometry comes from."""

    kind: str
    primitive: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    mesh: Mesh | None = None
    geometry: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    after: str | None = None
    # Content hash of ``mesh`` (computed once; a new mesh always means a new Source).
    digest: str | None = field(default=None, repr=False, compare=False)

    def to_json(self, mesh_index=None) -> dict[str, Any]:
        if self.kind == "primitive":
            return {
                "type": "primitive",
                "primitive": self.primitive,
                "params": value_to_json(self.params),
            }
        if self.kind == "mesh":
            return {"type": "mesh", "mesh": mesh_index(self.mesh) if mesh_index else None}
        data: dict[str, Any] = {
            "type": "geometry",
            "geometry": self.geometry,
            "parameters": value_to_json(self.overrides),
        }
        if self.after is not None:
            data["after"] = self.after
        return data


class Modifier:
    """One entry of a modifier stack (owned by a :class:`ProceduralGeometry`).

    Read ``id``, ``name``, ``type``, ``enabled`` and ``params`` (raw values:
    literals or :class:`Expr`). Change it with :meth:`set`, which is
    transactional like every procedural mutation.
    """

    def __init__(
        self,
        geometry: ProceduralGeometry,
        mid: str,
        mtype: str,
        name: str,
        enabled: bool,
        params: dict[str, Any],
    ) -> None:
        self._geometry = geometry
        geometry._system._handles[mid] = self
        self.id = mid
        self.type = mtype
        self.name = name
        self.enabled = enabled
        self.params = params

    def set(
        self, *, enabled: bool | None = None, name: str | None = None, **params: Any
    ) -> Modifier:
        """Change parameters (and/or ``enabled``/``name``); re-validated and re-evaluated."""
        self._geometry._system.set_modifier(
            self._geometry, self, enabled=enabled, name=name, params=params
        )
        return self

    def resolved(self) -> dict[str, Any]:
        """Parameter values after evaluating expressions and filling defaults."""
        system = self._geometry._system
        env = system._env(self._geometry, {})
        return system._resolve_modifier(self._geometry, self, env)[0]

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "params": value_to_json(self.params),
        }

    def __repr__(self) -> str:
        return f"Modifier({self.type!r}, name={self.name!r}, enabled={self.enabled})"


class ProceduralGeometry:
    """A reusable procedural geometry definition: source + parameters + modifier stack.

    Obtain one from ``scene.add_geometry(...)`` / ``scene.add_procedural(...)``.
    All mutating methods are transactional: invalid changes raise
    :class:`ProceduralError` and leave everything as it was.
    """

    def __init__(self, system: ProceduralSystem, gid: str, name: str, source: Source) -> None:
        self._system = system
        system._handles[gid] = self
        self.id = gid
        self.name = name
        self.source = source
        self.parameters: dict[str, Parameter] = {}
        self.rules: dict[str, Rule] = {}
        self.modifiers: list[Modifier] = []

    # -- parameters -----------------------------------------------------------

    def add_parameter(self, name: str, value: Any, **options: Any) -> Parameter:
        """Add a parameter (see :func:`~aether3d.procedural.params.make_parameter`)."""
        return self._system.add_parameter(name, value, geometry=self, **options)

    def set_parameter(self, name: str, value: Any) -> None:
        self._system.set_parameter(name, value, geometry=self)

    def remove_parameter(self, name: str) -> None:
        self._system.remove_parameter(name, geometry=self)

    def add_rule(self, name: str, expr: Any, message: str = "") -> Rule:
        """Forbid invalid combinations of this geometry's parameters (see ``add_rule``)."""
        return self._system.add_rule(name, expr, message, geometry=self)

    def remove_rule(self, name: str) -> None:
        self._system.remove_rule(name, geometry=self)

    def parameter_values(self, overrides: dict | None = None) -> dict[str, Any]:
        """Resolved values of this geometry's parameters (scene parameters included)."""
        env = self._system._env(self, self._system._resolve_overrides(self, overrides or {}))
        return {name: env[name] for name in self.parameters}

    # -- modifiers -----------------------------------------------------------

    def add_modifier(
        self,
        type: str,
        *,
        name: str | None = None,
        enabled: bool = True,
        index: int | None = None,
        **params: Any,
    ) -> Modifier:
        """Append (or insert at ``index``) a modifier of ``type`` with ``params``."""
        return self._system.add_modifier(
            self, type, name=name, enabled=enabled, index=index, params=params
        )

    def modifier(self, ref: str | int | Modifier) -> Modifier:
        """A modifier by id, name or stack index."""
        return self._system._find_modifier(self, ref)

    def remove_modifier(self, ref: str | int | Modifier) -> None:
        self._system.remove_modifier(self, ref)

    def move_modifier(self, ref: str | int | Modifier, index: int) -> None:
        self._system.move_modifier(self, ref, index)

    def set_source(self, **params: Any) -> None:
        """Change primitive parameters (or the overrides of a geometry-reference source)."""
        self._system.set_source(self, params)

    def convert_source_to_mesh(self) -> None:
        """Replace a primitive/reference source by its evaluated polygons as a stored base mesh."""
        self._system.convert_source_to_mesh(self)

    def edit_source(self):
        """Edit the stored base mesh with the mesh editing engine; ``commit()`` updates it.

        Only for ``mesh`` sources (use :meth:`convert_source_to_mesh` first for
        primitives). Every object using this geometry follows the change.
        """
        from .session import SourceEditSession

        return SourceEditSession(self)

    # -- evaluation ------------------------------------------------------------

    def evaluate(self, *, upto: str | int | None = None, overrides: dict | None = None) -> Mesh:
        """The evaluated triangle :class:`Mesh` (cached)."""
        return self._system.evaluate(self, upto=upto, overrides=overrides).mesh

    def evaluate_polygons(
        self, *, upto: str | int | None = None, overrides: dict | None = None
    ) -> PolyMesh:
        return self._system.evaluate(self, upto=upto, overrides=overrides).polygons

    def validate(self) -> dict[str, Any]:
        return self._system.validate(self)

    def info(self, *, evaluate: bool = False) -> dict[str, Any]:
        return self._system.info(self, evaluate=evaluate)

    @property
    def users(self) -> list[str]:
        """Ids of scene objects using this geometry."""
        return [o.id for o in self._system.scene if o._geometry == self.id]

    def __repr__(self) -> str:
        count = len(self.modifiers)
        return f"ProceduralGeometry(name={self.name!r}, id={self.id!r}, modifiers={count})"


@dataclass
class Evaluation:
    """Result of evaluating a geometry node."""

    key: str
    polygons: PolyMesh
    diagnostics: list[Diagnostic]
    _system: ProceduralSystem

    @property
    def mesh(self) -> Mesh:
        return self._system._final_mesh(self)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evaluations: dict[str, int] = field(default_factory=dict)  # step kind -> count computed

    def to_dict(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "evaluations": dict(self.evaluations)}


# --------------------------------------------------------------------------- #
# The system
# --------------------------------------------------------------------------- #


class ProceduralSystem:
    """Scene-level registry of procedural definitions (see the module docstring)."""

    cache_limit_bytes: int = 512 * 1024 * 1024
    max_history: int = 256

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.parameters: dict[str, Parameter] = {}
        self.rules: dict[str, Rule] = {}
        self._geometries: dict[str, ProceduralGeometry] = {}
        # Live handles by id, so undo/redo re-uses the objects callers still hold.
        self._handles: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        self._cache: OrderedDict[str, tuple[PolyMesh, list[Diagnostic]]] = OrderedDict()
        self._mesh_cache: OrderedDict[str, tuple[Mesh, list[Diagnostic]]] = OrderedDict()
        self._cache_bytes = 0
        self.stats = CacheStats()
        self._batch_depth = 0
        self._history: list[dict] = []
        self._redo: list[dict] = []
        self._revision = 0

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    @staticmethod
    def types() -> dict[str, Any]:
        """Schemas of every primitive and modifier type and the parameter types."""
        return {
            "primitives": [t.schema() for t in PRIMITIVES.values()],
            "modifiers": [t.schema() for t in MODIFIER_TYPES.values()],
            "parameter_types": list(PARAMETER_TYPES),
            "expression_functions": [
                "min",
                "max",
                "abs",
                "floor",
                "ceil",
                "round",
                "sqrt",
                "sin",
                "cos",
                "tan",
                "clamp",
                "int",
                "float",
            ],
        }

    @property
    def geometries(self) -> list[ProceduralGeometry]:
        return list(self._geometries.values())

    @property
    def revision(self) -> int:
        """Incremented by every successful change of procedural definitions."""
        return self._revision

    def geometry(self, ref: str | ProceduralGeometry) -> ProceduralGeometry:
        """A geometry node by id or name."""
        if isinstance(ref, ProceduralGeometry):
            if self._geometries.get(ref.id) is not ref:
                raise ProceduralError("UNKNOWN_GEOMETRY", f"{ref!r} does not belong to this scene")
            return ref
        if isinstance(ref, str):
            if ref in self._geometries:
                return self._geometries[ref]
            for g in self._geometries.values():
                if g.name == ref:
                    return g
        raise ProceduralError(
            "UNKNOWN_GEOMETRY",
            f"no geometry with id or name {ref!r}",
            hint=f"known: {', '.join(sorted(g.name for g in self.geometries)) or 'none'}",
        )

    # ------------------------------------------------------------------ #
    # Transactions and history
    # ------------------------------------------------------------------ #

    def _state(self) -> dict:
        return {
            "parameters": copy.deepcopy(self.parameters),
            "rules": copy.deepcopy(self.rules),
            "geometries": [
                {
                    "id": g.id,
                    "name": g.name,
                    # Base meshes are immutable and shared; everything else is copied.
                    "source": dataclasses.replace(
                        g.source,
                        params=copy.deepcopy(g.source.params),
                        overrides=copy.deepcopy(g.source.overrides),
                    ),
                    "parameters": copy.deepcopy(g.parameters),
                    "rules": copy.deepcopy(g.rules),
                    "modifiers": [
                        (m.id, m.type, m.name, m.enabled, copy.deepcopy(m.params))
                        for m in g.modifiers
                    ],
                }
                for g in self._geometries.values()
            ],
            "objects": {o.id: (o._geometry, o._mesh) for o in self.scene},
        }

    def _restore(self, state: dict) -> None:
        """Restore definitions *into the existing objects*, so handles stay valid."""
        self.parameters = _restore_parameters(self.parameters, state["parameters"])
        self.rules = copy.deepcopy(state["rules"])
        geometries = {}
        for data in state["geometries"]:
            g = self._handles.get(data["id"]) or ProceduralGeometry(
                self, data["id"], data["name"], data["source"]
            )
            g.name = data["name"]
            g.source = dataclasses.replace(
                data["source"],
                params=copy.deepcopy(data["source"].params),
                overrides=copy.deepcopy(data["source"].overrides),
            )
            g.parameters = _restore_parameters(g.parameters, data["parameters"])
            g.rules = copy.deepcopy(data["rules"])
            existing = {m.id: m for m in g.modifiers}
            modifiers = []
            for mid, mtype, name, enabled, params in data["modifiers"]:
                m = existing.get(mid) or self._handles.get(mid)
                if m is None:
                    m = Modifier(g, mid, mtype, name, enabled, {})
                m._geometry = g
                m.type, m.name, m.enabled, m.params = mtype, name, enabled, copy.deepcopy(params)
                modifiers.append(m)
            g.modifiers = modifiers
            geometries[g.id] = g
        self._geometries = geometries
        for obj in self.scene:
            saved = state["objects"].get(obj.id)
            # Objects that were and are static are not procedural history: leave their
            # meshes alone (they may have been edited since).
            if saved is not None and (saved[0] is not None or obj._geometry is not None):
                obj._geometry, obj._mesh = saved

    @contextmanager
    def batch(self):
        """Group several changes into one atomic, undoable transaction.

        Validation and evaluation run once at the end; if anything fails, every
        change made inside the block is rolled back.
        """
        with self._transaction():
            yield self

    @contextmanager
    def _transaction(self):
        if self._batch_depth:
            self._batch_depth += 1
            try:
                yield
            finally:
                self._batch_depth -= 1
            return
        before = self._state()
        self._batch_depth = 1
        try:
            yield
            self._batch_depth = 0
            self._check_all()
        except BaseException:
            self._batch_depth = 0
            self._restore(before)
            raise
        self._history.append(before)
        if len(self._history) > self.max_history:
            del self._history[0]
        self._redo.clear()
        self._revision += 1

    def undo(self) -> None:
        """Restore the procedural definitions as they were before the last change."""
        if not self._history:
            raise ProceduralError("NOTHING_TO_UNDO", "there is no procedural change to undo")
        self._step(self._history, self._redo)

    def redo(self) -> None:
        if not self._redo:
            raise ProceduralError("NOTHING_TO_REDO", "there is no undone procedural change to redo")
        self._step(self._redo, self._history)

    def _step(self, source: list[dict], target: list[dict]) -> None:
        current = self._state()
        self._restore(source[-1])
        try:
            self._check_all(evaluate=False)  # e.g. an object created since uses a node
        except ProceduralError as exc:
            self._restore(current)
            raise ProceduralError(
                "HISTORY_CONFLICT",
                f"cannot restore that state: {exc.message}",
                exc.location,
                hint="reassign or remove the objects that use geometry created since",
            ) from None
        source.pop()
        target.append(current)
        self._revision += 1

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ------------------------------------------------------------------ #
    # Mutations (all transactional)
    # ------------------------------------------------------------------ #

    def add_parameter(
        self, name: str, value: Any, *, geometry: Any = None, **options: Any
    ) -> Parameter:
        """Add a scene parameter, or a parameter of ``geometry``."""
        with self._transaction():
            scope = self.geometry(geometry).parameters if geometry is not None else self.parameters
            if name in scope:
                raise ProceduralError("DUPLICATE_PARAMETER", f"parameter {name!r} already exists")
            param = make_parameter(name, value, **options)
            scope[name] = param
            return param

    def set_parameter(self, name: str, value: Any, *, geometry: Any = None) -> None:
        """Set a parameter's value (literal or :class:`Expr`); affected geometry regenerates."""
        with self._transaction():
            scope = self.geometry(geometry).parameters if geometry is not None else self.parameters
            if name not in scope:
                raise ProceduralError(
                    "UNKNOWN_PARAMETER",
                    f"no parameter {name!r}",
                    hint=f"known: {', '.join(scope) or 'none'}",
                )
            scope[name].value = value_from_json(value) if isinstance(value, (dict, list)) else value

    def add_rule(self, name: str, expr: Any, message: str = "", *, geometry: Any = None) -> Rule:
        """Add a named condition that parameter values must satisfy.

        ``expr`` is a boolean expression over parameters in scope (the scene's,
        plus the geometry's own for geometry rules), e.g.
        ``"tower_count % 2 == 0"``. Rules are checked whenever values resolve,
        including with overrides, so invalid combinations are rejected with
        ``CONSTRAINT_VIOLATED`` and nothing changes.
        """
        with self._transaction():
            scope = self.geometry(geometry).rules if geometry is not None else self.rules
            if name in scope:
                raise ProceduralError("DUPLICATE_RULE", f"rule {name!r} already exists")
            rule = make_rule(name, expr, message)
            scope[name] = rule
            return rule

    def remove_rule(self, name: str, *, geometry: Any = None) -> None:
        with self._transaction():
            scope = self.geometry(geometry).rules if geometry is not None else self.rules
            if name not in scope:
                raise ProceduralError("UNKNOWN_RULE", f"no rule {name!r}")
            del scope[name]

    def set_parameters(self, values: dict[str, Any], *, geometry: Any = None) -> None:
        with self._transaction():
            for name, value in values.items():
                self.set_parameter(name, value, geometry=geometry)

    def remove_parameter(self, name: str, *, geometry: Any = None) -> None:
        with self._transaction():
            scope = self.geometry(geometry).parameters if geometry is not None else self.parameters
            if name not in scope:
                raise ProceduralError("UNKNOWN_PARAMETER", f"no parameter {name!r}")
            del scope[name]

    def add_geometry(
        self,
        name: str,
        *,
        primitive: str | None = None,
        params: dict[str, Any] | None = None,
        mesh: Mesh | None = None,
        source: Any = None,
        overrides: dict[str, Any] | None = None,
        after: str | int | None = None,
        geometry_id: str | None = None,
    ) -> ProceduralGeometry:
        """Create a geometry node from a primitive, a base mesh, or another geometry."""
        given = [x is not None for x in (primitive, mesh, source)]
        if sum(given) != 1:
            raise ProceduralError(
                "INVALID_SOURCE", "give exactly one of primitive=, mesh= or source="
            )
        with self._transaction():
            if not isinstance(name, str) or not name.strip():
                raise ProceduralError("INVALID_NAME", "geometry names must be non-empty strings")
            if any(g.name == name for g in self._geometries.values()):
                raise ProceduralError("DUPLICATE_NAME", f"a geometry named {name!r} already exists")
            gid = geometry_id or _new_id()
            if gid in self._geometries:
                raise ProceduralError("DUPLICATE_ID", f"geometry id {gid!r} already exists")
            if primitive is not None:
                if primitive not in PRIMITIVES:
                    raise ProceduralError(
                        "UNKNOWN_PRIMITIVE",
                        f"unknown primitive {primitive!r}",
                        hint=f"one of {', '.join(PRIMITIVES)}",
                    )
                src = Source("primitive", primitive=primitive, params=_values(params or {}))
            elif mesh is not None:
                if not isinstance(mesh, Mesh):
                    raise ProceduralError("INVALID_SOURCE", "mesh= must be an aether3d.Mesh")
                src = Source("mesh", mesh=mesh)
            else:
                ref = self.geometry(source)
                src = Source(
                    "geometry",
                    geometry=ref.id,
                    overrides=_values(overrides or {}),
                    after=None if after is None else self._find_modifier(ref, after).id,
                )
            g = ProceduralGeometry(self, gid, name, src)
            self._geometries[gid] = g
            return g

    def remove_geometry(self, ref: Any) -> None:
        with self._transaction():
            g = self.geometry(ref)
            users = g.users
            if users:
                raise ProceduralError(
                    "GEOMETRY_IN_USE",
                    f"geometry {g.name!r} is used by objects",
                    hint="bake or reassign those objects first",
                    details={"objects": users},
                )
            dependents = [d.name for d in self.dependents(g)]
            if dependents:
                raise ProceduralError(
                    "GEOMETRY_IN_USE",
                    f"geometry {g.name!r} is referenced by other geometry",
                    details={"geometry": dependents},
                )
            del self._geometries[g.id]

    def rename_geometry(self, ref: Any, name: str) -> None:
        with self._transaction():
            g = self.geometry(ref)
            if any(o.name == name and o is not g for o in self._geometries.values()):
                raise ProceduralError("DUPLICATE_NAME", f"a geometry named {name!r} already exists")
            if not isinstance(name, str) or not name.strip():
                raise ProceduralError("INVALID_NAME", "geometry names must be non-empty strings")
            g.name = name

    def set_source(self, ref: Any, params: dict[str, Any]) -> None:
        with self._transaction():
            g = self.geometry(ref)
            if g.source.kind == "primitive":
                g.source.params.update(_values(params))
            elif g.source.kind == "geometry":
                g.source.overrides.update(_values(params))
            else:
                raise ProceduralError(
                    "INVALID_SOURCE",
                    "a stored-mesh source has no parameters",
                    hint="use edit_source() to edit the base mesh",
                )

    def set_source_mesh(self, ref: Any, mesh: Mesh) -> None:
        with self._transaction():
            g = self.geometry(ref)
            if not isinstance(mesh, Mesh):
                raise ProceduralError("INVALID_SOURCE", "expected an aether3d.Mesh")
            g.source = Source("mesh", mesh=mesh)

    def convert_source_to_mesh(self, ref: Any) -> None:
        with self._transaction():
            g = self.geometry(ref)
            if g.source.kind == "mesh":
                return
            polygons = self._evaluate_source(g, self._env(g, {}), f"geometry[{g.name}].source")[1]
            mesh, _ = polygons.to_mesh()
            g.source = Source("mesh", mesh=mesh)

    def add_modifier(
        self,
        ref: Any,
        mtype: str,
        *,
        name: str | None = None,
        enabled: bool = True,
        index: int | None = None,
        params: dict | None = None,
    ) -> Modifier:
        with self._transaction():
            g = self.geometry(ref)
            if mtype not in MODIFIER_TYPES:
                raise ProceduralError(
                    "UNKNOWN_MODIFIER",
                    f"unknown modifier type {mtype!r}",
                    hint=f"one of {', '.join(MODIFIER_TYPES)}",
                )
            base = name or mtype.capitalize()
            existing = {m.name for m in g.modifiers}
            if name is not None and name in existing:
                raise ProceduralError(
                    "DUPLICATE_NAME", f"a modifier named {name!r} already exists in {g.name!r}"
                )
            unique, k = base, 1
            while unique in existing:
                unique, k = f"{base}.{k:03d}", k + 1
            if not isinstance(enabled, bool):
                raise ProceduralError("INVALID_PARAMETER", "enabled must be true or false")
            m = Modifier(
                g, _new_id(), mtype, unique, enabled, self._normalize_params(mtype, params or {})
            )
            if index is None:
                g.modifiers.append(m)
            else:
                if not isinstance(index, int) or not -len(g.modifiers) - 1 <= index <= len(
                    g.modifiers
                ):
                    raise ProceduralError("INVALID_INDEX", f"index {index!r} is out of range")
                g.modifiers.insert(index, m)
            return m

    def set_modifier(
        self,
        ref: Any,
        modifier: Any,
        *,
        enabled: bool | None = None,
        name: str | None = None,
        params: dict | None = None,
    ) -> None:
        with self._transaction():
            g = self.geometry(ref)
            m = self._find_modifier(g, modifier)
            if enabled is not None:
                if not isinstance(enabled, bool):
                    raise ProceduralError("INVALID_PARAMETER", "enabled must be true or false")
                m.enabled = enabled
            if name is not None:
                if any(o.name == name and o is not m for o in g.modifiers) or not str(name).strip():
                    raise ProceduralError(
                        "DUPLICATE_NAME", f"invalid or duplicate modifier name {name!r}"
                    )
                m.name = name
            if params:
                m.params.update(self._normalize_params(m.type, params))

    def remove_modifier(self, ref: Any, modifier: Any) -> None:
        with self._transaction():
            g = self.geometry(ref)
            m = self._find_modifier(g, modifier)
            g.modifiers.remove(m)

    def move_modifier(self, ref: Any, modifier: Any, index: int) -> None:
        with self._transaction():
            g = self.geometry(ref)
            m = self._find_modifier(g, modifier)
            if not isinstance(index, int) or not 0 <= index < len(g.modifiers):
                raise ProceduralError(
                    "INVALID_INDEX", f"index must be in 0..{len(g.modifiers) - 1}"
                )
            g.modifiers.remove(m)
            g.modifiers.insert(index, m)

    def assign(self, obj: Any, ref: Any | None) -> None:
        """Make a scene object use a geometry node (``None`` detaches, leaving no mesh)."""
        with self._transaction():
            g = None if ref is None else self.geometry(ref)
            obj._geometry = None if g is None else g.id
            if g is not None:
                obj._mesh = None

    def _normalize_params(self, mtype: str, params: dict) -> dict:
        out = _values(params)
        if mtype == "boolean" and isinstance(out.get("operand"), dict):
            operand = dict(out["operand"])
            if "geometry" in operand:
                operand["geometry"] = self.geometry(operand["geometry"]).id
            out["operand"] = operand
        return out

    def _find_modifier(self, g: ProceduralGeometry, ref: Any) -> Modifier:
        if isinstance(ref, Modifier):
            if ref in g.modifiers:
                return ref
        elif isinstance(ref, int) and not isinstance(ref, bool):
            if -len(g.modifiers) <= ref < len(g.modifiers):
                return g.modifiers[ref]
        elif isinstance(ref, str):
            for m in g.modifiers:
                if ref in (m.id, m.name):
                    return m
        raise ProceduralError(
            "UNKNOWN_MODIFIER",
            f"no modifier {ref!r} in geometry {g.name!r}",
            hint=f"modifiers: {', '.join(m.name for m in g.modifiers) or 'none'}",
        )

    # ------------------------------------------------------------------ #
    # Dependencies
    # ------------------------------------------------------------------ #

    def _references(self, g: ProceduralGeometry) -> list[str]:
        refs = []
        if g.source.kind == "geometry":
            refs.append(g.source.geometry)
        for m in g.modifiers:
            refs += MODIFIER_TYPES[m.type].references(m.params)
        return refs

    def dependencies(self, ref: Any) -> list[ProceduralGeometry]:
        """Geometry nodes this node reads (directly or indirectly), in evaluation order."""
        g = self.geometry(ref)
        order: list[str] = []

        def visit(node: str) -> None:
            for r in self._references(self._geometries[node]):
                if r not in order and r in self._geometries:
                    visit(r)
                    order.append(r)

        visit(g.id)
        return [self._geometries[i] for i in order]

    def dependents(self, ref: Any) -> list[ProceduralGeometry]:
        """Geometry nodes that read this node (directly or indirectly)."""
        g = self.geometry(ref)
        return [o for o in self._geometries.values() if o is not g and g in self.dependencies(o)]

    def parameter_dependencies(self, ref: Any = None) -> dict[str, list[str]]:
        """Which parameters each parameter's expression reads (scene or geometry scope)."""
        scope = self.geometry(ref).parameters if ref is not None else self.parameters
        return {
            name: sorted(p.value.names) if isinstance(p.value, Expr) else []
            for name, p in scope.items()
        }

    def _check_cycles(self) -> None:
        state: dict[str, int] = {}

        def visit(node: str, path: list[str]) -> None:
            state[node] = 1
            for r in self._references(self._geometries[node]):
                if r not in self._geometries:
                    raise ProceduralError(
                        "UNKNOWN_GEOMETRY",
                        f"geometry {self._geometries[node].name!r} "
                        f"references missing geometry {r!r}",
                        f"geometry[{node}]",
                    )
                if state.get(r) == 1:
                    cycle = [self._geometries[n].name for n in path[path.index(r) :]] + [
                        self._geometries[r].name
                    ]
                    raise ProceduralError(
                        "DEPENDENCY_CYCLE",
                        "geometry references form a cycle: " + " -> ".join(cycle),
                        details={"cycle": cycle},
                    )
                if r not in state:
                    visit(r, [*path, r])
            state[node] = 2

        for node in self._geometries:
            if node not in state:
                visit(node, [node])

    # ------------------------------------------------------------------ #
    # Validation of definitions (run after every transaction)
    # ------------------------------------------------------------------ #

    def _check_all(self, *, evaluate: bool = True) -> None:
        self._check_cycles()
        for obj in self.scene:
            if obj._geometry is not None and obj._geometry not in self._geometries:
                raise ProceduralError(
                    "UNKNOWN_GEOMETRY", f"object {obj.name!r} uses missing geometry"
                )
        self._env(None, {})  # scene parameters resolve (types, ranges, cycles)
        for g in self._geometries.values():
            for m in g.modifiers:
                env = self._env(g, {})
                self._resolve_modifier(g, m, env)
        # Evaluate every node, so failures surface at the change that caused them
        # (cached: unchanged nodes cost a key computation only).
        if evaluate:
            for g in self._geometries.values():
                self.evaluate(g)

    # ------------------------------------------------------------------ #
    # Parameter resolution
    # ------------------------------------------------------------------ #

    def _resolve_scope(
        self,
        params: dict[str, Parameter],
        base_env: dict,
        location: str,
        raw_override: dict | None = None,
    ) -> dict:
        env = dict(base_env)
        raw = {name: p.value for name, p in params.items()}
        if raw_override:
            raw.update(raw_override)
        resolving: list[str] = []
        done: dict[str, Any] = {}

        def resolve(name: str) -> Any:
            if name in done:
                return done[name]
            if name in resolving:
                cycle = resolving[resolving.index(name) :] + [name]
                raise ProceduralError(
                    "DEPENDENCY_CYCLE",
                    "parameter expressions form a cycle: " + " -> ".join(cycle),
                    f"{location}.{name}",
                    details={"cycle": cycle},
                )
            resolving.append(name)
            value = raw[name]
            if isinstance(value, Expr):
                local = {**env, **{n: resolve(n) for n in value.names if n in raw}}
                try:
                    value = evaluate_expression(value, local)
                except ProceduralError as exc:
                    raise ProceduralError(exc.code, exc.message, f"{location}.{name}") from None
            p = params[name]
            try:
                value = check_type(p.type, value, minimum=p.min, maximum=p.max, choices=p.choices)
            except ProceduralError as exc:
                raise ProceduralError(exc.code, exc.message, f"{location}.{name}") from None
            resolving.pop()
            done[name] = value
            return value

        for name in raw:
            if name not in params:
                raise ProceduralError(
                    "UNKNOWN_PARAMETER", f"no parameter {name!r} to override", location
                )
            env[name] = resolve(name)
        return env

    def _env(self, g: ProceduralGeometry | None, overrides: dict) -> dict[str, Any]:
        scene_env = self._resolve_scope(self.parameters, {}, "parameters")
        check_rules(self.rules, scene_env, "parameters")
        if g is None:
            return scene_env
        location = f"geometry[{g.name}].parameters"
        env = self._resolve_scope(g.parameters, scene_env, location, overrides)
        check_rules(g.rules, env, location)
        return env

    def _resolve_overrides(self, g: ProceduralGeometry, overrides: dict) -> dict:
        unknown = sorted(set(overrides) - set(g.parameters))
        if unknown:
            raise ProceduralError(
                "UNKNOWN_PARAMETER", f"geometry {g.name!r} has no parameter(s) {', '.join(unknown)}"
            )
        return _values(overrides)

    def _resolve_values(self, raw: dict, env: dict, location: str) -> dict:
        def resolve(value: Any, loc: str) -> Any:
            if isinstance(value, Expr):
                try:
                    return evaluate_expression(value, env)
                except ProceduralError as exc:
                    raise ProceduralError(exc.code, exc.message, loc) from None
            if isinstance(value, tuple):
                return tuple(resolve(v, loc) for v in value)
            if isinstance(value, dict):
                return {k: resolve(v, f"{loc}.{k}") for k, v in value.items()}
            return value

        return {k: resolve(v, f"{location}.{k}") for k, v in raw.items()}

    def _resolve_modifier(self, g: ProceduralGeometry, m: Modifier, env: dict) -> tuple[dict, str]:
        location = f"geometry[{g.name}].modifiers[{m.name}]"
        mtype = MODIFIER_TYPES[m.type]
        raw = dict(m.params)
        operand_overrides = None
        if isinstance(raw.get("operand"), dict) and isinstance(
            raw["operand"].get("parameters"), dict
        ):
            operand_overrides = raw["operand"]["parameters"]
            raw["operand"] = {k: v for k, v in raw["operand"].items() if k != "parameters"}
        values = self._resolve_values(raw, env, f"{location}.params")
        if operand_overrides is not None:
            values["operand"]["parameters"] = self._resolve_values(
                operand_overrides, env, f"{location}.params.operand.parameters"
            )
        values = validate_values(mtype.params, values, f"{location}.params")
        mtype.check(values, f"{location}.params")
        return values, location

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self, ref: Any, *, upto: str | int | None = None, overrides: dict | None = None
    ) -> Evaluation:
        """Evaluate a geometry node (cached); raises ProceduralError with a location on failure."""
        g = self.geometry(ref)
        resolved = self._resolve_overrides(g, overrides or {})
        key, polygons, diagnostics = self._evaluate(g, resolved, upto, ())
        return Evaluation(key, polygons, diagnostics, self)

    def mesh(self, ref: Any) -> Mesh:
        return self.evaluate(ref).mesh

    def _evaluate(self, g: ProceduralGeometry, overrides: dict, upto: Any, stack: tuple) -> tuple:
        if g.id in stack:  # defensive: cycles are rejected at definition time
            raise ProceduralError("DEPENDENCY_CYCLE", f"geometry {g.name!r} depends on itself")
        stack = (*stack, g.id)
        env = self._env(g, overrides)
        location = f"geometry[{g.name}]"
        key, polygons, diagnostics = self._evaluate_source(g, env, f"{location}.source", stack)
        diagnostics = list(diagnostics)
        stop = None if upto is None else self._find_modifier(g, upto)
        for m in g.modifiers:
            if m.enabled:
                values, mloc = self._resolve_modifier(g, m, env)
                operands: dict[str, PolyMesh] = {}
                operand_keys = []
                mtype = MODIFIER_TYPES[m.type]
                for ref in mtype.references(values):
                    other = self.geometry(ref)
                    over = values["operand"].get("parameters", {})
                    okey, opoly, _ = self._evaluate(
                        other, self._resolve_overrides(other, over), None, stack
                    )
                    operands[ref] = opoly
                    operand_keys.append(okey)
                key = _key("modifier", key, m.type, values, operand_keys)

                def compute(m=m, values=values, mloc=mloc, operands=operands, polygons=polygons):
                    ctx = ModifierContext(mloc, lambda spec, loc: operands[spec["geometry"]])
                    try:
                        out, diags = MODIFIER_TYPES[m.type].apply(polygons, values, ctx)
                    except ProceduralError:
                        raise
                    except Exception as exc:  # a backend or numerical failure: report, don't leak
                        raise ProceduralError(
                            "MODIFIER_FAILED", f"{type(exc).__name__}: {exc}", mloc
                        ) from exc
                    _check_finite(out, mloc)
                    return out, diags

                polygons, mdiags = self._cached(key, m.type, compute)
                diagnostics += mdiags
            if stop is m:
                break
        return key, polygons, diagnostics

    def _evaluate_source(
        self, g: ProceduralGeometry, env: dict, location: str, stack: tuple = ()
    ) -> tuple:
        src = g.source
        if src.kind == "primitive":
            values = self._resolve_values(src.params, env, f"{location}.params")
            key = _key("primitive", src.primitive, values)

            def build():
                return PRIMITIVES[src.primitive].generate(values, f"{location}.params"), []

            polygons, diags = self._cached(key, "primitive", build)
            return key, polygons, diags
        if src.kind == "mesh":
            if src.digest is None:
                src.digest = mesh_hash(src.mesh)
            key = _key("mesh", src.digest)
            polygons, diags = self._cached(key, "mesh", lambda: (PolyMesh.from_mesh(src.mesh), []))
            return key, polygons, diags
        other = self.geometry(src.geometry)
        over = self._resolve_overrides(
            other, self._resolve_values(src.overrides, env, f"{location}.parameters")
        )
        return self._evaluate(other, over, src.after, stack)

    def _cached(self, key: str, kind: str, compute) -> tuple[PolyMesh, list[Diagnostic]]:
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return hit
        self.stats.misses += 1
        self.stats.evaluations[kind] = self.stats.evaluations.get(kind, 0) + 1
        value = compute()
        self._cache[key] = value
        self._cache_bytes += _size(value[0])
        self._evict()
        return value

    def _final_mesh(self, evaluation: Evaluation) -> Mesh:
        hit = self._mesh_cache.get(evaluation.key)
        if hit is not None:
            self._mesh_cache.move_to_end(evaluation.key)
            self.stats.hits += 1
            return hit[0]
        self.stats.misses += 1
        self.stats.evaluations["mesh"] = self.stats.evaluations.get("mesh", 0) + 1
        try:
            mesh, diags = evaluation.polygons.to_mesh()
        except PolyMeshError as exc:
            raise ProceduralError("TRIANGULATION_FAILED", str(exc)) from None
        self._mesh_cache[evaluation.key] = (mesh, diags)
        self._cache_bytes += _mesh_size(mesh)
        self._evict()
        return mesh

    def _evict(self) -> None:
        while self._cache_bytes > self.cache_limit_bytes and (
            len(self._cache) > 1 or len(self._mesh_cache) > 1
        ):
            if len(self._cache) > 1:
                _, (poly, _) = self._cache.popitem(last=False)
                self._cache_bytes -= _size(poly)
            else:
                _, (mesh, _) = self._mesh_cache.popitem(last=False)
                self._cache_bytes -= _mesh_size(mesh)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._mesh_cache.clear()
        self._cache_bytes = 0

    @property
    def cache_info(self) -> dict[str, Any]:
        return {
            "entries": len(self._cache),
            "meshes": len(self._mesh_cache),
            "bytes": self._cache_bytes,
            "limit_bytes": self.cache_limit_bytes,
            **self.stats.to_dict(),
        }

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #

    def validate(self, ref: Any = None) -> dict[str, Any]:
        """Evaluate and check one geometry node (or all); never raises for geometry problems."""
        from ..inspection import inspect_mesh

        targets = [self.geometry(ref)] if ref is not None else self.geometries
        reports = []
        for g in targets:
            entry: dict[str, Any] = {"geometry": {"id": g.id, "name": g.name}}
            try:
                evaluation = self.evaluate(g)
                mesh = evaluation.mesh
            except ProceduralError as exc:
                entry.update(status="error", error=exc.to_dict())
                reports.append(entry)
                continue
            topology = evaluation.polygons.validate()
            inspection = inspect_mesh(mesh)
            triangulation = self._mesh_cache.get(evaluation.key, (None, []))[1]
            diagnostics = evaluation.diagnostics + list(triangulation) + topology.diagnostics
            entry.update(
                status=worst_severity(diagnostics + inspection.diagnostics),
                topology=topology.to_dict(),
                inspection={
                    "status": inspection.status,
                    "topology": inspection.topology,
                    "surface_area": inspection.surface_area,
                    "bounds": inspection.bounds,
                },
                diagnostics=[d.to_dict() for d in evaluation.diagnostics + list(triangulation)],
            )
            reports.append(entry)
        status = "ok"
        for r in reports:
            if r["status"] == "error":
                status = "error"
            elif r["status"] == "warning" and status == "ok":
                status = "warning"
        return {
            "status": status,
            "geometry": reports,
            "not_evaluated": ["self-intersection between different faces"],
        }

    def info(self, ref: Any = None, *, evaluate: bool = False) -> dict[str, Any]:
        """Structured description of definitions (and optionally evaluated statistics)."""
        if ref is None:
            scene_env = self._env(None, {})
            return {
                "revision": self._revision,
                "parameters": {
                    name: {**p.to_json(), "resolved": value_to_json(scene_env[name])}
                    for name, p in self.parameters.items()
                },
                "rules": [r.to_json() for r in self.rules.values()],
                "geometry": [self.info(g, evaluate=evaluate) for g in self.geometries],
                "cache": self.cache_info,
            }
        g = self.geometry(ref)
        env = self._env(g, {})
        data: dict[str, Any] = {
            "id": g.id,
            "name": g.name,
            "source": g.source.to_json(lambda m: "<mesh>"),
            "parameters": {
                name: {**p.to_json(), "resolved": value_to_json(env[name])}
                for name, p in g.parameters.items()
            },
            "rules": [r.to_json() for r in g.rules.values()],
            "modifiers": [
                {**m.info(), "resolved": value_to_json(self._resolve_modifier(g, m, env)[0])}
                for m in g.modifiers
            ],
            "dependencies": [d.name for d in self.dependencies(g)],
            "dependents": [d.name for d in self.dependents(g)],
            "users": g.users,
        }
        if g.source.kind == "mesh":
            data["source"]["mesh"] = {
                "vertex_count": g.source.mesh.vertex_count,
                "triangle_count": g.source.mesh.triangle_count,
            }
        if evaluate:
            evaluation = self.evaluate(g)
            summary = evaluation.polygons.validate().summary
            mesh = evaluation.mesh
            data["evaluated"] = {
                "key": evaluation.key,
                "polygons": summary,
                "mesh": {"vertex_count": mesh.vertex_count, "triangle_count": mesh.triangle_count},
                "diagnostics": [d.to_dict() for d in evaluation.diagnostics],
            }
        return data


def _restore_parameters(current: dict[str, Parameter], saved: dict[str, Parameter]) -> dict:
    out = {}
    for name, snapshot in saved.items():
        param = current.get(name) or copy.deepcopy(snapshot)
        for f in dataclasses.fields(Parameter):
            setattr(param, f.name, copy.deepcopy(getattr(snapshot, f.name)))
        out[name] = param
    return out


def _values(params: dict) -> dict:
    if not isinstance(params, dict):
        raise ProceduralError("INVALID_PARAMETER", "parameters must be given as a mapping")
    return {k: value_from_json(v) if isinstance(v, (dict, list)) else v for k, v in params.items()}


def _check_finite(mesh: PolyMesh, location: str) -> None:
    if not np.all(np.isfinite(mesh.positions)):
        raise ProceduralError(
            "NON_FINITE_RESULT",
            "the modifier produced non-finite positions",
            location,
            hint="check for extreme or degenerate parameter values",
        )


def _size(mesh: PolyMesh) -> int:
    total = mesh.positions.nbytes + mesh.face_offsets.nbytes + mesh.face_vertices.nbytes
    for arr in (mesh.uvs, mesh.normals):
        if arr is not None:
            total += arr.nbytes
    return total


def _mesh_size(mesh: Mesh) -> int:
    total = mesh.positions.nbytes + mesh.triangles.nbytes
    for arr in (mesh.uvs, mesh.normals):
        if arr is not None:
            total += arr.nbytes
    return total


__all__ = ["Evaluation", "Modifier", "ProceduralGeometry", "ProceduralSystem", "Source"]
