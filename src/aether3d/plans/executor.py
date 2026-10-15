"""Executing scene plans: bind to a scene, check everything, then build in bulk.

``execute_plan(scene, plan)`` runs in three phases:

1. **Parse** (:mod:`.model`): the whole document is checked statically.
2. **Bind**: references are resolved against the scene, meshes are built (pure
   computation, shared by content), patterns are expanded into flat object records,
   names are chosen and checked, ids are derived, limits are checked. Nothing has
   changed yet; any problem raises :class:`~aether3d.errors.PlanError`.
3. **Apply**: one history transaction (or, opt-in, one per chunk). Materials,
   textures and procedural geometry are created through their APIs; objects through
   the bulk path ``Scene._create_many``, which records each parent's child list once.
   Then cameras and lights, aiming, operations, world and settings, framing. If
   anything fails -- or the caller cancels, or the time is up -- the transaction
   restores the scene exactly and ``PlanError`` names the failing part.

``mode="validate"`` stops after binding (and reports estimates); ``mode="dry_run"``
applies the plan and rolls it back, reporting exactly what it would do.
"""

from __future__ import annotations

import contextlib
import math
import re
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError, PlanError
from ..math3d import quat, quat_from_euler
from ..mesh import Mesh
from ..procedural.params import Expr
from ..scene import NewObject, SceneObject
from .limits import PlanLimits
from .model import (
    _BAD,
    MeshSpec,
    ObjectSpec,
    OperationSpec,
    Plan,
    SceneRef,
    Selector,
    SymbolRef,
    evaluate,
    parse_plan,
    render_name,
)
from .patterns import Instances, expand, vary

if TYPE_CHECKING:
    from ..scene import Scene

MODES = ("execute", "dry_run", "validate")
ID_NAMESPACE = uuid.UUID("6f2b7a52-5d1e-4c7b-9a0e-3e2d8c4b1a90")
"""Namespace of the object ids derived from idempotency keys (``uuid5``)."""
AUTO_IDS = 20
"""``result_ids: "auto"`` lists the objects of definitions that made at most this many."""
_SLICE = 4096  # objects created per bulk call (cancellation and progress points)
_INDEX_RE = re.compile(r"\[-?\d+\]")

# Plans applied with an idempotency key, per scene (in-process; see SCENE_PLANS.md).
_APPLIED: weakref.WeakKeyDictionary[Scene, dict[str, dict[str, Any]]] = weakref.WeakKeyDictionary()


class _Cancelled(Exception):
    pass


class _DryRun(Exception):
    """Raised inside the transaction of a dry run to roll it back."""


@dataclass
class PlanResult:
    """What a plan did (or, validated, what it would do). ``to_dict()`` is its JSON."""

    ok: bool
    mode: str
    status: str  # applied | dry_run | validated | already_applied | invalid
    label: str
    fingerprint: str
    idempotency_key: str | None = None
    revision: dict[str, int] = field(default_factory=dict)
    step: dict[str, Any] | None = None
    counts: dict[str, Any] = field(default_factory=dict)
    symbols: dict[str, Any] = field(default_factory=dict)
    operations: list[dict[str, Any]] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    estimate: dict[str, Any] | None = None
    chunks: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ok": self.ok,
            "mode": self.mode,
            "status": self.status,
            "label": self.label,
            "fingerprint": self.fingerprint,
            "idempotency_key": self.idempotency_key,
            "revision": dict(self.revision),
            "step": self.step,
            "counts": dict(self.counts),
            "symbols": self.symbols,
            "operations": self.operations,
            "queries": self.queries,
            "warnings": self.warnings,
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
        }
        if self.issues:
            data["issues"] = self.issues
        if self.estimate is not None:
            data["estimate"] = self.estimate
        if self.chunks is not None:
            data["chunks"] = self.chunks
        return data


@dataclass
class _Constants:
    """What a definition's objects share (see ``_Run._constants``)."""

    kind: str = "empty"
    mesh: Mesh | None = None
    geometry: Any = None
    material: Any = _BAD
    color: Any = None
    specular: Any = None
    shininess: Any = None
    appearance: Any = None
    appearance_varies: bool = False
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    visible: bool = True
    camera: dict | None = None
    light: dict | None = None
    camera_value: Any = None
    light_value: Any = None
    group_materials: dict[int, Any] = field(default_factory=dict)
    aims: bool = False
    children: list[ObjectSpec] = field(default_factory=list)
    group: str = ""  # the definition's path without instance indices ("houses/walls")


@dataclass
class _Layout:
    """The instances of one definition under one parent (see ``_Run._layout``)."""

    inst: Instances
    count: int
    varied: dict[str, list]
    rows: list[dict[str, Any]] | None
    envs: list[dict[str, Any]] | None
    variables: dict[str, list] = field(default_factory=dict)
    positions: list[tuple] = field(default_factory=list)
    rotations: Any = None  # one quaternion, or one per instance
    scales: Any = None  # one scale, or one per instance
    indices: list[int] = field(default_factory=list)  # pattern index of each instance

    def subset(self, keep: list[int]) -> None:
        """Keep only these instances (a ``where`` filter); everything stays aligned."""

        def pick(values: Any) -> Any:
            return [values[i] for i in keep] if isinstance(values, list) else values

        self.positions = pick(self.positions)
        self.rotations = pick(self.rotations)
        self.scales = pick(self.scales)
        self.varied = {f: pick(v) for f, v in self.varied.items()}
        self.rows = pick(self.rows)
        self.envs = pick(self.envs)
        self.variables = {k: pick(v) for k, v in self.variables.items()}
        self.indices = keep
        self.count = len(keep)


@dataclass
class _Record:
    """One object to create: the bulk record plus what the plan still has to do to it."""

    new: NewObject
    spec: ObjectSpec
    path: str  # instance path: "towers[3]/roof"
    parent: int | SceneObject | None
    material: Any = _BAD  # a MaterialSpec symbol (str) to resolve after creation
    group_materials: dict[int, Any] = field(default_factory=dict)
    geometry: Any = None  # a geometry symbol (str) or an existing node id
    camera: dict | None = None
    light: dict | None = None
    aim: tuple[str, Any, Any] | None = None  # ("at", target, up) | ("along", direction, up)
    created: SceneObject | None = None


def execute_plan(
    scene: Scene,
    plan: Any,
    *,
    mode: str = "execute",
    expected_revision: int | None = None,
    limits: PlanLimits | None = None,
    base_dir: str | Path | None = None,
    trusted_paths: bool = False,
    progress: Callable[[int, int, str], Any] | None = None,
    cancel: Any = None,
    timeout: float | None = None,
) -> PlanResult:
    """Execute a scene plan (a dict or JSON text); see ``docs/SCENE_PLANS.md``.

    * ``mode`` -- ``"execute"`` (default), ``"dry_run"`` (apply, report, roll back) or
      ``"validate"`` (check and estimate without building).
    * ``expected_revision`` -- refuse (``REVISION_MISMATCH``) unless ``scene.revision``
      is this (optimistic concurrency); also accepted in the plan.
    * ``limits`` -- a :class:`PlanLimits`.
    * ``base_dir`` -- where texture image paths resolve (default: the project's
      directory); they may not leave it unless ``trusted_paths``.
    * ``progress(done, total, message)`` -- called at safe points of long plans.
    * ``cancel`` -- an object with ``is_set()`` (``threading.Event``) or a callable;
      when it turns true the plan stops at the next safe point, rolls back and raises
      ``PlanError("CANCELLED")``.
    * ``timeout`` -- seconds; past it the plan rolls back (``TIMEOUT``).

    Raises :class:`~aether3d.errors.PlanError` (the scene is then unchanged, except for
    the completed chunks of a chunked plan, see ``options.chunk_size``).
    """
    if mode not in MODES:
        raise PlanError("INVALID_VALUE", f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    start = time.perf_counter()
    limits = limits or PlanLimits()
    parsed = parse_plan(plan, limits)
    run = _Run(
        scene,
        parsed,
        mode=mode,
        limits=limits,
        base_dir=base_dir,
        trusted_paths=trusted_paths,
        progress=progress,
        cancel=cancel,
        timeout=timeout if timeout is not None else parsed.options.timeout,
        start=start,
    )
    if expected_revision is not None:
        if parsed.expected_revision is not None and parsed.expected_revision != expected_revision:
            raise PlanError(
                "INVALID_VALUE",
                f"expected_revision is {expected_revision} in the call and "
                f"{parsed.expected_revision} in the plan",
                "/expected_revision",
            )
        run.expected_revision = expected_revision
    run.timings["parse"] = time.perf_counter() - start
    return run.run()


def validate_plan(
    plan: Any,
    scene: Scene | None = None,
    *,
    limits: PlanLimits | None = None,
    base_dir: str | Path | None = None,
) -> PlanResult:
    """Check a plan without building anything; never raises for a problem in the plan.

    Returns a :class:`PlanResult` with ``ok``, ``issues`` (every problem found, with
    JSON pointers) and ``estimate`` (objects, meshes, triangles, memory). With a
    ``scene`` the references, names and selections are checked against it too.
    """
    try:
        if scene is None:
            return _static_validation(plan, limits or PlanLimits())
        return execute_plan(scene, plan, mode="validate", limits=limits, base_dir=base_dir)
    except PlanError as exc:
        label = plan.get("label", "Scene plan") if isinstance(plan, dict) else "Scene plan"
        issues = exc.issues or [
            {
                "code": exc.code,
                "message": exc.message,
                "pointer": exc.pointer or "",
                "operation": exc.operation,
                "hint": exc.hint,
                "details": exc.details,
            }
        ]
        return PlanResult(
            ok=False,
            mode="validate",
            status="invalid",
            label=label if isinstance(label, str) else "Scene plan",
            fingerprint="",
            issues=issues,
        )


def select_objects(scene: Scene, selector: Any, pointer: str = "/select") -> list[SceneObject]:
    """The objects of ``scene`` a plan selector names, in scene order.

    ``selector`` is what a plan operation's ``select`` takes: an id or name, a list of
    them, or an object with ``ids``, ``names``, ``tags``, ``any_tags``, ``name_glob``,
    ``kind``, ``under``, ``visible``, ``material`` and ``expect`` (plan symbols such as
    ``"@towers"`` exist only inside a plan). Raises :class:`PlanError`, located at
    ``pointer`` (where the selector came from).
    """
    from .model import Issues, Reader, parse_selector

    issues = Issues()
    sel = parse_selector(Reader(issues, PlanLimits()), selector, pointer)
    issues.raise_if_any()
    if sel is None:  # pragma: no cover - an issue was recorded
        raise PlanError("INVALID_VALUE", "invalid selector", pointer)
    for k, ref in enumerate(sel.refs):
        if isinstance(ref, SymbolRef):
            raise PlanError(
                "INVALID_REFERENCE",
                f"{ref.text!r}: plan symbols exist only inside a plan",
                f"{pointer}/refs/{k}",
                hint="name objects by id or name here",
            )
    run = _Run(
        scene,
        parse_plan({"format": "aether3d.scene-plan", "version": 1}),
        mode="validate",
        limits=PlanLimits(),
        base_dir=None,
        trusted_paths=False,
        progress=None,
        cancel=None,
        timeout=None,
        start=time.perf_counter(),
    )
    return run.select(sel, pointer)


def with_selections(scene: Scene, request: Any, pointer: str = "") -> Any:
    """A spatial query request whose object lists may be selectors.

    ``objects``, ``against`` and ``exclude`` take ids or names; here they may also be a
    selector object as :func:`select_objects` takes (``{"tags": ["tree"]}``,
    ``{"under": "Keep"}``, ``{"name_glob": "Tower */*"}``), which becomes the list of the
    objects it selects (possibly empty: then the query sees none). Raises
    :class:`PlanError` for a malformed selector or an unmet ``expect``, located under
    ``pointer`` (the request's own location).
    """
    from ..spatial.request import OBJECT_LISTS

    if not isinstance(request, dict):
        return request
    out = dict(request)
    for key in OBJECT_LISTS:
        if isinstance(out.get(key), dict):
            out[key] = [o.id for o in select_objects(scene, out[key], f"{pointer}/{key}")]
    return out


def _static_validation(plan: Any, limits: PlanLimits) -> PlanResult:
    """Check a plan without a scene: its structure, and how many objects it makes."""
    start = time.perf_counter()
    parsed = parse_plan(plan, limits)

    filtered = []

    def count(spec: ObjectSpec, depth: int = 0) -> int:
        n = spec.pattern.count if spec.pattern is not None else 1
        if spec.where is not None:
            filtered.append(spec.path)
        children = list(spec.children)
        if spec.kind == "prefab" and isinstance(spec.prefab, SymbolRef):
            prefab = parsed.prefabs.get(spec.prefab.head)
            if prefab is not None and depth < limits.max_depth:
                children += prefab.content.children
        return n * (1 + sum(count(c, depth + 1) for c in children))

    by_symbol = {spec.symbol: count(spec) for spec in parsed.objects}
    total = sum(by_symbol.values())
    if total > limits.max_objects:
        raise PlanError(
            "LIMIT_EXCEEDED",
            f"the plan creates {total} objects (limit {limits.max_objects})",
            "/objects",
        )
    return PlanResult(
        ok=True,
        mode="validate",
        status="validated",
        label=parsed.label,
        fingerprint=parsed.fingerprint,
        idempotency_key=parsed.idempotency_key,
        warnings=[
            {
                "code": "NOT_BOUND",
                "message": "checked without a scene: references to existing objects, "
                "materials and geometry, names and selections were not checked",
                "pointer": "",
                "details": {},
            }
        ],
        estimate={
            "objects": total,
            "objects_at_most": bool(filtered),  # "where" filters skip some instances
            "by_symbol": by_symbol,
            "meshes": len(parsed.meshes),
            "materials": len(parsed.materials),
            "textures": len(parsed.textures),
            "geometries": len(parsed.geometries),
            "operations": len(parsed.operations),
            "queries": len(parsed.queries),
        },
        timings={"total": time.perf_counter() - start},
    )


def _material_error(exc: AetherError) -> dict[str, Any]:
    if hasattr(exc, "to_dict"):
        try:
            return {"type": type(exc).__name__, **exc.to_dict()}
        except Exception:  # pragma: no cover - defensive
            pass
    return {"type": type(exc).__name__, "message": str(exc)}


class _Run:
    def __init__(
        self,
        scene: Scene,
        plan: Plan,
        *,
        mode: str,
        limits: PlanLimits,
        base_dir: Any,
        trusted_paths: bool,
        progress: Any,
        cancel: Any,
        timeout: float | None,
        start: float,
    ) -> None:
        self.scene = scene
        self.plan = plan
        self.mode = mode
        self.limits = limits
        root = base_dir if base_dir is not None else scene.materials.base_dir or Path.cwd()
        self.base_dir = Path(root).absolute()
        self.trusted_paths = trusted_paths
        self.progress_fn = progress
        self.cancel = cancel
        self.deadline = None if timeout is None else start + float(timeout)
        self.start = start
        self.expected_revision = plan.expected_revision
        self.timings: dict[str, float] = {}
        self.warnings: list[dict[str, Any]] = []
        self.meshes: dict[str, Mesh] = {}  # mesh symbol -> mesh
        self._mesh_cache: dict[str, Mesh] = {}  # content key -> mesh (sharing)
        self._borrowed_meshes: set[int] = set()  # ids of scene meshes (from_object)
        self._mesh_users: dict[int, int] = {}  # id(mesh) -> objects using it
        self.records: list[_Record] = []
        self.by_path: dict[str, list[int]] = {}  # spec path -> record indices (in order)
        self.instances: dict[str, list[int]] = {}  # instance path -> record index
        self.materials: dict[str, Any] = {}  # material symbol -> Material
        self.textures: dict[str, Any] = {}
        self.geometries: dict[str, Any] = {}  # geometry symbol -> ProceduralGeometry
        self.op_results: list[dict[str, Any]] = []
        self.query_results: list[dict[str, Any]] = []
        self.created_ids: set[str] = set()
        self._done = 0
        self._total = 1
        self._last_progress = 0.0
        self._appearances: dict[tuple, Any] = {}
        self._compiled: dict[int, _Constants] = {}
        self._layouts: dict[int, _Layout] = {}
        self.groups: dict[str, list[str]] = {}  # "houses/walls" -> its spec paths

    # ------------------------------------------------------------------ #
    # Safe points
    # ------------------------------------------------------------------ #

    def checkpoint(self, done: int = 0, message: str = "") -> None:
        """A safe point: honour cancellation and the deadline, report progress."""
        self._done += done
        cancel = self.cancel
        if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
            raise PlanError(
                "CANCELLED",
                "the plan was cancelled; nothing was changed",
                hint="the scene is as it was before the plan",
            )
        now = time.perf_counter()
        if self.deadline is not None and now > self.deadline:
            raise PlanError(
                "TIMEOUT",
                f"the plan did not finish within its timeout ({self.deadline - self.start:g} s); "
                "nothing was changed",
                hint="raise the timeout, split the plan, or use options.chunk_size",
            )
        if self.progress_fn is not None and (message or now - self._last_progress > 0.1):
            self._last_progress = now
            self.progress_fn(min(self._done, self._total), self._total, message)

    def warn(self, code: str, message: str, pointer: str = "", **details: Any) -> None:
        if len(self.warnings) < 100:
            self.warnings.append(
                {"code": code, "message": message, "pointer": pointer, "details": details}
            )

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #

    def run(self) -> PlanResult:
        scene, plan = self.scene, self.plan
        revision_before = scene.revision
        if self.expected_revision is not None and self.expected_revision != revision_before:
            raise PlanError(
                "REVISION_MISMATCH",
                f"the scene is at revision {revision_before}, the plan expects "
                f"{self.expected_revision}",
                "/expected_revision",
                hint="read the scene again (it changed since), then resend the plan",
                details={"revision": revision_before, "expected": self.expected_revision},
            )
        previous = self._previous_application()
        if previous is not None:
            return previous
        t = time.perf_counter()
        try:
            self.bind()
        except _AlreadyApplied:
            # Every object of the plan (ids derived from its idempotency key) is in the
            # scene already: an earlier run applied it, maybe in another process.
            for rec in self.records:
                rec.created = scene._objects[rec.new.object_id]
            result = self._result("already_applied", revision_before, None)
            self._remember(result)
            return result
        self.timings["bind"] = time.perf_counter() - t
        if self.mode == "validate":
            return self._result("validated", revision_before, None)
        chunks = None
        if (
            self.mode == "execute"
            and plan.options.chunk_size is not None
            and len(self.records) > plan.options.chunk_size
        ):
            chunks = self._chunks()
        t = time.perf_counter()
        history = None
        if chunks is not None:
            chunk_report = self.apply_chunked(chunks)
            self.timings["apply"] = time.perf_counter() - t
            result = self._result("applied", revision_before, chunk_report.get("step"))
            result.chunks = chunk_report
            self._remember(result)
            return result
        try:
            with scene.history.transaction(plan.label) as tx:
                self.apply(range(len(self.records)), first=True, last=True)
                if self.mode == "dry_run":
                    self.run_queries()
                    self.timings["apply"] = time.perf_counter() - t
                    raise _DryRun
        except _DryRun:
            result = self._result("dry_run", revision_before, None)
            result.revision["after"] = scene.revision
            return result
        self.timings["apply"] = time.perf_counter() - t
        if tx.entry is not None:
            history = tx.entry.to_dict()
            history.pop("paths", None)
        t = time.perf_counter()
        self.run_queries()
        self.timings["queries"] = time.perf_counter() - t
        result = self._result("applied", revision_before, history)
        self._remember(result)
        return result

    def _result(self, status: str, revision_before: int, history: Any) -> PlanResult:
        plan = self.plan
        self.timings["total"] = time.perf_counter() - self.start
        result = PlanResult(
            ok=True,
            mode=self.mode,
            status=status,
            label=plan.label,
            fingerprint=plan.fingerprint,
            idempotency_key=plan.idempotency_key,
            revision={"before": revision_before, "after": self.scene.revision},
            step=history,
            counts=self._counts(),
            symbols=self._symbols(),
            operations=self.op_results,
            queries=self.query_results,
            warnings=self.warnings,
            timings=dict(self.timings),
        )
        if status == "validated":
            result.estimate = self._estimate()
        return result

    # ------------------------------------------------------------------ #
    # Idempotency
    # ------------------------------------------------------------------ #

    def _previous_application(self) -> PlanResult | None:
        """The result of an earlier run of this plan (same idempotency key), if any."""
        key = self.plan.idempotency_key
        if key is None or self.mode != "execute":
            return None
        applied = _APPLIED.get(self.scene, {}).get(key)
        if applied is None:
            return None
        if applied["fingerprint"] != self.plan.fingerprint:
            raise PlanError(
                "IDEMPOTENCY_KEY_REUSED",
                f"idempotency key {key!r} was used for a different plan",
                "/idempotency_key",
                hint="use a new key for a new plan; resend the same plan to retry it",
            )
        ids = applied["object_ids"]
        present = sum(1 for i in ids if i in self.scene._objects)
        if ids and present != len(ids):
            raise PlanError(
                "IDEMPOTENCY_CONFLICT",
                f"the plan with key {key!r} was applied, but {len(ids) - present} of its "
                f"{len(ids)} objects are no longer in the scene (undone or removed)",
                "/idempotency_key",
                hint="use a new idempotency key to apply the plan again",
                details={"present": present, "objects": len(ids)},
            )
        result = PlanResult(**{**applied["result"].__dict__})
        result.status = "already_applied"
        result.timings = {"total": time.perf_counter() - self.start}
        return result

    def _remember(self, result: PlanResult) -> None:
        key = self.plan.idempotency_key
        if key is None:
            return
        _APPLIED.setdefault(self.scene, {})[key] = {
            "fingerprint": self.plan.fingerprint,
            "object_ids": [r.created.id for r in self.records if r.created is not None],
            "result": result,
        }

    def _object_id(self, path: str, spec: ObjectSpec) -> str | None:
        if spec.object_id is not None:
            return spec.object_id
        key = self.plan.idempotency_key
        if key is None:
            return None
        return str(uuid.uuid5(ID_NAMESPACE, f"{key}\x1f{path}"))

    # ------------------------------------------------------------------ #
    # Binding
    # ------------------------------------------------------------------ #

    def bind(self) -> None:
        plan = self.plan
        self.env = dict(plan.parameters)
        self._geometry_order()  # cycles are reported before anything is built
        for sym, spec in plan.meshes.items():
            self.meshes[sym] = self.build_mesh(spec)
        order = self._object_order()
        for spec in order:
            self.expand_spec(spec, parent=None, parent_name=None, prefix="", top=True)
            self.checkpoint()
        self._check_ids()  # an earlier run of this plan (same idempotency key)?
        self._check_resources()
        self._choose_names()
        self._check_limits()
        self._total = len(self.records) + len(plan.operations) + 4

    def _check_resources(self) -> None:
        """Names of new materials, textures and geometry nodes must be free.

        Resources an earlier, interrupted run of the same plan created (their ids are
        derived from the idempotency key) are not conflicts: they are reused.
        """
        library = self.scene.materials
        for kind, specs, taken in (
            ("material", self.plan.materials, library._material_names),
            ("texture", self.plan.textures, library._texture_names),
        ):
            names: set[str] = set()
            for sym, spec in specs.items():
                existing = taken.get(spec.name)
                if existing is not None and existing.id == self._resource_id(kind, sym):
                    names.add(spec.name)
                    continue
                if spec.name in taken or spec.name in names:
                    raise PlanError(
                        "NAME_CONFLICT",
                        f"a {kind} named {spec.name!r} already exists",
                        f"{spec.pointer}/name" if spec.name != spec.symbol else spec.pointer,
                        hint=f'reference the existing {kind} by name (without "@"), or give '
                        "this one another name",
                    )
                names.add(spec.name)
        if len(self.plan.materials) + len(self.plan.textures) > self.limits.max_materials:
            raise PlanError(
                "LIMIT_EXCEEDED",
                f"the plan creates more than {self.limits.max_materials} materials and textures",
                "/materials",
            )
        system = self.scene._procedural
        existing = set() if system is None else {g.name for g in system.geometries}
        names = set()
        resuming = getattr(self, "present_ids", 0) > 0
        for spec in self.plan.geometries.values():
            if (spec.name in existing and not resuming) or spec.name in names:
                raise PlanError(
                    "NAME_CONFLICT",
                    f"a geometry named {spec.name!r} already exists",
                    spec.pointer,
                    hint='reference the existing geometry by name (without "@"), or rename',
                )
            names.add(spec.name)

    def _geometry_order(self) -> list[str]:
        plan = self.plan
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(sym: str, path: list[str]) -> None:
            state[sym] = 1
            spec = plan.geometries[sym]
            for ref in spec.references():
                if isinstance(ref, SymbolRef):
                    head = ref.head
                    if state.get(head) == 1:
                        cycle = [*path[path.index(head) :], head]
                        raise PlanError(
                            "REFERENCE_CYCLE",
                            "geometries reference each other in a cycle: " + " -> ".join(cycle),
                            spec.pointer,
                            details={"cycle": cycle},
                        )
                    if head not in state:
                        visit(head, [*path, head])
            state[sym] = 2
            order.append(sym)

        for sym in plan.geometries:
            if sym not in state:
                visit(sym, [sym])
        return order

    def _object_order(self) -> list[ObjectSpec]:
        """Top-level objects with plan parents after those parents (cycles refused)."""
        specs = {s.symbol: s for s in self.plan.objects}
        order: list[ObjectSpec] = []
        state: dict[str, int] = {}

        def visit(spec: ObjectSpec, path: list[str]) -> None:
            state[spec.symbol] = 1
            parent = spec.parent
            if isinstance(parent, SymbolRef):
                head = parent.head
                if state.get(head) == 1:
                    cycle = [*path[path.index(head) :], head]
                    raise PlanError(
                        "REFERENCE_CYCLE",
                        "objects are each other's parents: " + " -> ".join(cycle),
                        f"{spec.pointer}/parent",
                        details={"cycle": cycle},
                    )
                if head in specs and head not in state:
                    visit(specs[head], [*path, head])
            state[spec.symbol] = 2
            order.append(spec)

        for spec in self.plan.objects:
            if spec.symbol not in state:
                visit(spec, [spec.symbol])
        self._check_prefab_cycles()
        return order

    def _check_prefab_cycles(self) -> None:
        prefabs = self.plan.prefabs

        def uses(spec: ObjectSpec) -> list[str]:
            out = [spec.prefab.head] if isinstance(spec.prefab, SymbolRef) else []
            for child in spec.children:
                out += uses(child)
            return out

        state: dict[str, int] = {}

        def visit(sym: str, path: list[str]) -> None:
            state[sym] = 1
            for used in uses(prefabs[sym].content):
                if used not in prefabs:
                    continue
                if state.get(used) == 1:
                    cycle = [*path[path.index(used) :], used]
                    raise PlanError(
                        "REFERENCE_CYCLE",
                        "prefabs contain each other: " + " -> ".join(cycle),
                        prefabs[sym].pointer,
                        details={"cycle": cycle},
                    )
                if used not in state:
                    visit(used, [*path, used])
            state[sym] = 2

        for sym in prefabs:
            if sym not in state:
                visit(sym, [sym])

    # -- meshes ------------------------------------------------------------------

    def build_mesh(self, spec: MeshSpec) -> Mesh:
        """A mesh for a definition; identical definitions share one mesh."""
        key = None
        if spec.primitive is not None:
            params = {
                k: evaluate(v, self.env, f"{spec.pointer}/{k}") for k, v in spec.params.items()
            }
            key = repr((spec.primitive, sorted(params.items())))
            cached = self._mesh_cache.get(key)
            if cached is not None:
                return cached
            mesh = _primitive_mesh(spec.primitive, params, spec.pointer, self.limits)
        elif spec.data is not None:
            try:
                data = spec.data
                mesh = Mesh(
                    np.asarray(data["positions"], dtype=np.float64),
                    np.asarray(data["triangles"], dtype=np.int64),
                    normals=None if data.get("normals") is None else np.asarray(data["normals"]),
                    uvs=None if data.get("uvs") is None else np.asarray(data["uvs"]),
                    groups=None if data.get("groups") is None else np.asarray(data["groups"]),
                )
            except (AetherError, ValueError, TypeError) as exc:
                raise PlanError(
                    "INVALID_VALUE", f"invalid mesh data: {exc}", f"{spec.pointer}/data"
                ) from None
        else:
            ref = spec.from_object
            if isinstance(ref, SymbolRef):
                raise PlanError(
                    "INVALID_REFERENCE",
                    "from_object takes an existing object; plan objects share meshes by symbol",
                    f"{spec.pointer}/from_object",
                )
            obj = self.scene_object(ref, f"{spec.pointer}/from_object")
            mesh = obj.mesh
            if mesh is None:
                raise PlanError(
                    "INVALID_REFERENCE",
                    f"object {obj.name!r} has no mesh",
                    f"{spec.pointer}/from_object",
                )
            if obj._mesh is mesh:  # shared with the scene: costs no new memory
                self._borrowed_meshes.add(id(mesh))
            return mesh
        if key is not None:
            self._mesh_cache[key] = mesh
        return mesh

    # -- references to the scene -------------------------------------------------

    def scene_object(self, ref: Any, pointer: str) -> SceneObject:
        scene = self.scene
        if isinstance(ref, SceneRef):
            obj = None
            if ref.by in ("any", "id"):
                obj = scene._objects.get(ref.value)
            if obj is None and ref.by in ("any", "name"):
                obj = scene._by_name.get(ref.value)
            if obj is None:
                raise PlanError(
                    "UNKNOWN_REFERENCE",
                    f"no object with {'id or name' if ref.by == 'any' else ref.by} {ref.value!r} "
                    "in the scene",
                    pointer,
                    hint='plan objects are referenced as "@symbol"',
                )
            return obj
        objects = self.resolve_objects(ref, pointer)
        if len(objects) != 1:
            raise PlanError(
                "AMBIGUOUS_REFERENCE",
                f"{ref.text!r} names {len(objects)} objects; one is needed here",
                pointer,
                hint=f"pick one instance: {ref.text}[0]",
            )
        return objects[0]

    def resolve_paths(self, ref: SymbolRef, pointer: str) -> list[int]:
        """Record indices of the objects a symbol reference names."""
        current: list[str] = []  # instance paths
        for depth, (name, index) in enumerate(ref.path):
            base = [name] if depth == 0 else [f"{p}/{name}" for p in current]
            matched: list[str] = []
            for b in base:
                indices = self.by_path.get(b)
                if indices is None:
                    continue
                paths = [self.records[i].path for i in indices]
                if index is not None:
                    if not -len(paths) <= index < len(paths):
                        raise PlanError(
                            "UNKNOWN_REFERENCE",
                            f"{ref.text!r}: index {index} is outside 0..{len(paths) - 1}",
                            pointer,
                        )
                    paths = [paths[index]]
                matched += paths
            if not matched:
                section = self.plan.symbols.get(name) if depth == 0 else None
                if section == "prefabs":
                    hint = (
                        f"{name!r} is a prefab, not an object: reach its copies through the "
                        f"objects that use it (@object/child), or select them by tag"
                    )
                elif section is not None and section != "objects":
                    hint = f"{name!r} is defined in {section}, not in objects"
                else:
                    hint = "children are reached as @parent/child, instances as @symbol[index]"
                raise PlanError(
                    "UNKNOWN_REFERENCE",
                    f"{ref.text!r} names no object of this plan",
                    pointer,
                    hint=hint,
                )
            current = matched
        return [self.instances[p] for p in current]

    def resolve_objects(self, ref: Any, pointer: str) -> list[SceneObject]:
        if isinstance(ref, SceneRef):
            return [self.scene_object(ref, pointer)]
        indices = self.resolve_paths(ref, pointer)
        objects = [self.records[i].created for i in indices]
        if any(o is None for o in objects):
            raise PlanError(
                "INVALID_REFERENCE",
                f"{ref.text!r} is not created yet at this point of the plan",
                pointer,
            )
        return objects  # type: ignore[return-value]

    def material_id(self, ref: Any, pointer: str) -> str | None:
        """The id of a material reference: an existing one now, a plan one after creation."""
        if ref is None:
            return None
        if isinstance(ref, SymbolRef):
            material = self.materials.get(ref.head)
            if material is None:
                raise PlanError("INVALID_REFERENCE", f"{ref.text!r} is not created yet", pointer)
            return material.id
        found = self.scene.materials.find_material(ref.value)
        if (
            found is None
            or (ref.by == "id" and found.id != ref.value)
            or (ref.by == "name" and found.name != ref.value)
        ):
            raise PlanError(
                "UNKNOWN_REFERENCE",
                f"no material {ref.value!r} in the scene",
                pointer,
                hint='plan materials are referenced as "@symbol"',
            )
        return found.id

    def geometry_node(self, ref: Any, pointer: str) -> Any:
        if isinstance(ref, SymbolRef):
            return ref.head  # created in the apply phase
        system = self.scene._procedural
        try:
            if system is None:
                raise KeyError(ref.value)
            return system.geometry(ref.value).id
        except (AetherError, KeyError):
            raise PlanError(
                "UNKNOWN_REFERENCE",
                f"no geometry {ref.value!r} in the scene",
                pointer,
                hint='plan geometries are referenced as "@symbol"',
            ) from None

    # -- expanding objects ----------------------------------------------------------

    def _content(self, spec: ObjectSpec) -> ObjectSpec:
        """What an object is: its own definition, or the prefab it instantiates."""
        if spec.kind != "prefab":
            return spec
        if not isinstance(spec.prefab, SymbolRef):
            raise PlanError(
                "INVALID_REFERENCE", 'prefabs are plan symbols ("@name")', f"{spec.pointer}/prefab"
            )
        return self.plan.prefabs[spec.prefab.head].content

    def expand_spec(
        self,
        spec: ObjectSpec,
        *,
        parent: int | SceneObject | None,
        parent_name: str | None,
        prefix: str,
        top: bool = False,
    ) -> None:
        """Records for every instance of ``spec`` and, recursively, of its children.

        What does not depend on the instance is computed once per definition (cached
        across the parents that repeat it, as prefab children do); positions, rotations
        and scales are combined as arrays.
        """
        if top and spec.parent is not None:
            parent, parent_name = self._top_parent(spec)
        spec_path = f"{prefix}{spec.symbol}"
        c = self._constants(spec)
        lay = self._layout(spec, spec_path, c)
        n = lay.count
        if len(self.records) + n > self.limits.max_objects:
            raise PlanError(
                "LIMIT_EXCEEDED",
                f"the plan creates more than {self.limits.max_objects} objects",
                spec.pointer,
                hint="split the plan, or raise the object limit (PlanLimits.max_objects)",
                details={"limit": self.limits.max_objects},
            )
        ptr = spec.pointer
        patterned = spec.pattern is not None
        varied, rows = lay.varied, lay.rows
        indices = self.by_path.setdefault(spec_path, [])
        self.groups.setdefault(_INDEX_RE.sub("", spec_path), []).append(spec_path)
        template = spec.name
        default_base = spec.symbol if parent_name is None or top else f"{parent_name}/{spec.symbol}"
        name_values = {"n": lay.inst.count, "symbol": spec.symbol, "parent": parent_name or ""}
        grid = lay.variables
        records = self.records
        positions, scales, rotations = lay.positions, lay.scales, lay.rotations
        for i in range(n):  # i: this definition's i-th created instance
            path = f"{spec_path}[{i}]" if patterned else spec_path
            index_i = lay.indices[i]  # its pattern index (names, expressions)
            row = rows[i] if rows is not None else None
            # name
            row_template = row.get("name") if row else None
            if row_template is not None:
                name = render_name(row_template, {**name_values, "i": index_i})
            elif template is not None:
                name = render_name(
                    template,
                    {
                        **name_values,
                        "i": index_i,
                        "x": grid["x"][i] if "x" in grid else 0,
                        "y": grid["y"][i] if "y" in grid else 0,
                        "z": grid["z"][i] if "z" in grid else 0,
                    },
                )
                if patterned and not template.placeholders:
                    name = f"{name} {index_i}"
            else:
                name = f"{default_base} {index_i}" if patterned else default_base
            # per-instance overrides
            material = c.material
            appearance = c.appearance
            visible = c.visible
            tags = c.tags
            metadata = c.metadata
            mesh, geometry = c.mesh, c.geometry
            if varied:
                if "material" in varied:
                    material = varied["material"][i]
                if "mesh" in varied:
                    mesh = self._mesh_of(varied["mesh"][i], f"{ptr}/vary/mesh")
                if "geometry" in varied:
                    geometry = self.geometry_node(varied["geometry"][i], f"{ptr}/vary/geometry")
                if "visible" in varied:
                    visible = bool(varied["visible"][i])
                if "tags" in varied:
                    tags = tags + tuple(varied["tags"][i])
            if row:
                if "material" in row:
                    material = row["material"]
                if "visible" in row:
                    visible = row["visible"]
                if "tags" in row:
                    tags = tags + tuple(row["tags"])
                if "metadata" in row:
                    metadata = {**(metadata or {}), **row["metadata"]}
            if c.appearance_varies:
                color = varied["color"][i] if "color" in varied else c.color
                if row and "color" in row:
                    color = row["color"]
                appearance = self._appearance(
                    color,
                    varied["specular"][i] if "specular" in varied else c.specular,
                    varied["shininess"][i] if "shininess" in varied else c.shininess,
                    material,
                    ptr,
                )
            elif appearance is not None and material is not _BAD and material is not None:
                self._appearance(c.color, c.specular, c.shininess, material, ptr)  # refuses
            new = NewObject(
                name,
                mesh=mesh,
                camera=c.camera_value,
                light=c.light_value,
                position=positions[i],
                rotation=rotations[i] if isinstance(rotations, list) else rotations,
                scale=scales[i] if isinstance(scales, list) else scales,
                object_id=self._object_id(path, spec),
                visible=visible,
                appearance=appearance,
                tags=tags,
                metadata=metadata,
            )
            rec = _Record(new, spec, path, parent)
            if material is not _BAD:
                rec.material = material
            rec.group_materials = c.group_materials
            rec.geometry = geometry
            rec.camera, rec.light = c.camera, c.light
            if c.aims:
                rec.aim = self._aim_of(
                    spec, lay.envs[i] if lay.envs else self._instance_env(lay.inst, i, n)
                )
            if mesh is not None:
                self._mesh_users[id(mesh)] = self._mesh_users.get(id(mesh), 0) + 1
            index = len(records)
            records.append(rec)
            indices.append(index)
            self.instances[path] = index
            for child in c.children:
                self.expand_spec(child, parent=index, parent_name=name, prefix=f"{path}/")

    def _constants(self, spec: ObjectSpec) -> _Constants:
        """Everything about ``spec``'s objects that does not depend on the instance."""
        found = self._compiled.get(id(spec))
        if found is not None:
            return found
        content = self._content(spec)
        own = spec is content
        c = _Constants()
        kind = content.kind
        ptr = spec.pointer
        c.kind = kind
        if kind == "mesh" and "mesh" not in spec.vary:
            c.mesh = self._mesh_of(content.mesh, f"{content.pointer}/mesh")
        if kind == "geometry" and "geometry" not in spec.vary:
            c.geometry = self.geometry_node(content.geometry, f"{content.pointer}/geometry")
        if kind not in ("mesh", "geometry") and any(k in spec.vary for k in ("mesh", "geometry")):
            raise PlanError("INVALID_VALUE", f"a {kind} object has no mesh to vary", f"{ptr}/vary")
        if content.material_given:
            c.material = content.material
        if not own and spec.material_given:
            c.material = spec.material
        c.color = spec.color if spec.color is not None else content.color
        c.specular = spec.specular if spec.specular is not None else content.specular
        c.shininess = spec.shininess if spec.shininess is not None else content.shininess
        c.tags = tuple(content.tags) + (() if own else tuple(spec.tags)) + self.plan.options.tags
        c.metadata = content.metadata
        if not own and spec.metadata is not None:
            c.metadata = {**(c.metadata or {}), **spec.metadata}
        c.visible = bool(spec.visible and content.visible)
        c.camera = dict(content.camera or {}) if kind == "camera" else None
        c.light = dict(content.light or {}) if kind == "light" else None
        if c.camera is not None:
            c.camera_value = self._component("camera", c.camera, f"{content.pointer}/camera")
        if c.light is not None:
            c.light_value = self._component("light", c.light, f"{content.pointer}/light")
        if kind in ("camera", "light") and (c.material is not _BAD or c.color is not None):
            raise PlanError("INVALID_VALUE", f"a {kind} object has no material or color", ptr)
        c.group_materials = dict(content.group_materials)
        if not own:
            c.group_materials.update(spec.group_materials)
        rows = spec.pattern.params.get("rows") if spec.pattern is not None else None
        c.appearance_varies = any(k in spec.vary for k in ("color", "specular", "shininess")) or (
            rows is not None and any("color" in r for r in rows)
        )
        if not c.appearance_varies:
            c.appearance = self._appearance(c.color, c.specular, c.shininess, _BAD, ptr)
            if c.appearance is not None and c.material is not _BAD and c.material is not None:
                self._appearance(c.color, c.specular, c.shininess, c.material, ptr)  # refuses
        c.aims = (
            spec.look_at is not None or spec.look_along is not None or spec.direction is not None
        )
        c.children = list(content.children) + ([] if own else list(spec.children))
        c.group = spec.path
        self._compiled[id(spec)] = c
        return c

    @staticmethod
    def _component(kind: str, params: dict[str, Any], pointer: str) -> Any:
        from ..rendering.camera import Camera
        from ..rendering.light import Light

        try:
            return Camera(**params) if kind == "camera" else Light(**params)
        except (AetherError, TypeError) as exc:
            raise PlanError(
                "INVALID_VALUE", getattr(exc, "message", None) or str(exc), pointer
            ) from None

    def _layout(self, spec: ObjectSpec, spec_path: str, c: _Constants) -> _Layout:
        """Instances of ``spec``: placement and variations (cached when not random)."""
        random = (spec.pattern is not None and spec.pattern.kind == "scatter") or any(
            g.kind in ("choose", "uniform") for g in spec.vary.values()
        )
        if not random:
            found = self._layouts.get(id(spec))
            if found is not None:
                return found
        ptr = spec.pointer
        inst = expand(spec.pattern, self.env, spec_path)
        self.warnings.extend(inst.warnings)
        n = inst.count
        dynamic = _uses_instance(spec.position, spec.rotation_euler, spec.scale, spec.up)
        dynamic = dynamic or (
            c.aims and _uses_instance(spec.look_at, spec.look_along, spec.direction)
        )
        if dynamic or spec.where is not None or any(g.kind == "expr" for g in spec.vary.values()):
            envs = [self._instance_env(inst, i, n) for i in range(n)]
        else:
            envs = None
        varied = {f: vary(g, f, n, envs or [], spec_path) for f, g in spec.vary.items()}
        rows = inst.rows

        def evaluated(value: Any, what: str) -> Any:
            env = envs[0] if envs else self._instance_env(inst, 0, n)
            return evaluate(value, env, f"{ptr}/{what}")

        # positions
        positions = inst.positions
        if spec.position is not None:
            if _uses_instance(spec.position):
                own = [_vec3(evaluate(spec.position, e, f"{ptr}/position"), ptr) for e in envs]
                positions = positions + np.array(own)
            else:
                positions = positions + np.array(_vec3(evaluated(spec.position, "position"), ptr))
        if "offset" in varied:
            positions = positions + np.array(varied["offset"], dtype=np.float64)
        # rotations (Euler degrees, summed), then quaternions
        euler = None
        if spec.rotation_euler is not None:
            if _uses_instance(spec.rotation_euler):
                euler = np.array(
                    [
                        _vec3(evaluate(spec.rotation_euler, e, f"{ptr}/rotation_euler"), ptr)
                        for e in envs
                    ]
                )
            else:
                euler = np.array(_vec3(evaluated(spec.rotation_euler, "rotation_euler"), ptr))
        extras = [inst.rotations]
        if rows is not None and any("rotation_euler" in r for r in rows):
            extras.append(np.array([r.get("rotation_euler", (0.0, 0.0, 0.0)) for r in rows]))
        if "rotation_euler" in varied:
            extras.append(np.array(varied["rotation_euler"], dtype=np.float64))
        for extra in extras:
            if extra is not None:
                euler = extra.astype(np.float64) if euler is None else euler + extra
        if spec.rotation is not None and euler is not None:
            raise PlanError(
                "INVALID_VALUE",
                "this object's rotation varies per instance; give rotation_euler, not a quaternion",
                f"{ptr}/rotation",
            )
        # scales
        scales: Any = None
        if spec.scale is not None:
            if _uses_instance(spec.scale):
                scales = np.array(
                    [_scale3(evaluate(spec.scale, e, f"{ptr}/scale"), ptr) for e in envs]
                )
            else:
                scales = np.array(_scale3(evaluated(spec.scale, "scale"), ptr))
        factors = [inst.scales]
        if rows is not None and any("scale" in r for r in rows):
            factors.append(np.array([r.get("scale", (1.0, 1.0, 1.0)) for r in rows]))
        if "scale" in varied:
            factors.append(np.array(varied["scale"], dtype=np.float64))
        for factor in factors:
            if factor is not None:
                scales = factor.astype(np.float64) if scales is None else scales * factor
        if not np.all(np.isfinite(positions)) or (
            scales is not None and not np.all(np.isfinite(scales))
        ):
            raise PlanError("INVALID_VALUE", "an instance has a non-finite position or scale", ptr)
        keep = None
        if spec.where is not None:
            keep = []
            for i, env in enumerate(envs):
                value = evaluate(spec.where, env, f"{ptr}/where")
                if not isinstance(value, bool):
                    raise PlanError(
                        "INVALID_VALUE",
                        f"where must be true or false, got {value!r}",
                        f"{ptr}/where",
                    )
                if value:
                    keep.append(i)
        lay = _Layout(inst=inst, count=n, varied=varied, rows=rows, envs=envs)
        lay.variables = inst.variables
        lay.positions = [tuple(p) for p in np.broadcast_to(positions, (n, 3)).tolist()]
        if scales is None:
            lay.scales = (1.0, 1.0, 1.0)
        elif scales.ndim == 1:
            lay.scales = tuple(float(v) for v in scales)
        else:
            lay.scales = [tuple(v) for v in scales.tolist()]
        if spec.rotation is not None:
            lay.rotations = quat(spec.rotation)
        elif euler is None:
            lay.rotations = (0.0, 0.0, 0.0, 1.0)
        elif euler.ndim == 1:
            if not np.all(np.isfinite(euler)):
                raise PlanError("INVALID_VALUE", "a rotation is not finite", ptr)
            lay.rotations = quat_from_euler(tuple(float(v) for v in euler))
        else:
            if not np.all(np.isfinite(euler)):
                raise PlanError("INVALID_VALUE", "an instance has a non-finite rotation", ptr)
            memo: dict[tuple, Any] = {}
            rotations = []
            for row in euler.tolist():
                key = tuple(row)
                q = memo.get(key)
                if q is None:
                    q = memo[key] = quat_from_euler(key)
                rotations.append(q)
            lay.rotations = rotations
        lay.indices = list(range(n))
        if keep is not None and len(keep) < n:
            lay.subset(keep)
        if not random:
            self._layouts[id(spec)] = lay
        return lay

    def _aim_of(self, spec: ObjectSpec, env: dict[str, Any]) -> tuple[str, Any, Any]:
        ptr = spec.pointer
        up = None if spec.up is None else _vec3(evaluate(spec.up, env, f"{ptr}/up"), f"{ptr}/up")
        if spec.look_at is not None:
            target = spec.look_at
            if not isinstance(target, (SymbolRef, SceneRef)):
                target = _vec3(evaluate(target, env, f"{ptr}/look_at"), f"{ptr}/look_at")
            return ("at", target, up)
        value = spec.look_along if spec.look_along is not None else spec.direction
        return ("along", _vec3(evaluate(value, env, f"{ptr}/look_along"), ptr), up)

    def _top_parent(self, spec: ObjectSpec) -> tuple[Any, str | None]:
        ref = spec.parent
        pointer = f"{spec.pointer}/parent"
        if isinstance(ref, SceneRef):
            obj = self.scene_object(ref, pointer)
            if obj.camera is not None or obj.light is not None:
                raise PlanError(
                    "INVALID_REFERENCE", "cameras and lights cannot be parents", pointer
                )
            return obj, obj.name
        indices = self.resolve_paths(ref, pointer)
        if len(indices) != 1:
            raise PlanError(
                "AMBIGUOUS_REFERENCE",
                f"{ref.text!r} names {len(indices)} objects; a parent is one object",
                pointer,
                hint='give every instance its own children with "children" instead',
            )
        rec = self.records[indices[0]]
        if rec.camera is not None or rec.light is not None:
            raise PlanError("INVALID_REFERENCE", "cameras and lights cannot be parents", pointer)
        return indices[0], rec.new.name

    def _instance_env(self, inst: Instances, i: int, n: int) -> dict[str, Any]:
        env = dict(self.env)
        variables = inst.variables
        env.update(
            i=i,
            n=n,
            u=i / (n - 1) if n > 1 else 0.0,
            x=variables["x"][i] if "x" in variables else 0,
            y=variables["y"][i] if "y" in variables else 0,
            z=variables["z"][i] if "z" in variables else 0,
            angle=variables["angle"][i] if "angle" in variables else 0.0,
        )
        return env

    def _mesh_of(self, source: Any, pointer: str) -> Mesh:
        if isinstance(source, MeshSpec):
            return self.build_mesh(source)
        if isinstance(source, SymbolRef):
            return self.meshes[source.head]
        obj = self.scene_object(source, pointer)
        if obj.mesh is None:
            raise PlanError("INVALID_REFERENCE", f"object {obj.name!r} has no mesh", pointer)
        return obj.mesh

    def _appearance(
        self, color: Any, specular: Any, shininess: Any, material: Any, pointer: str
    ) -> Any:
        """A shared, validated appearance for these values (``None`` if all are unset)."""
        from ..rendering.settings import Appearance

        if color is None and specular is None and shininess is None:
            return None
        if material is not _BAD and material is not None:
            raise PlanError(
                "INVALID_VALUE",
                "an object with a material takes its color from the material",
                f"{pointer}/color",
                hint="set the color on the material, or leave material out",
            )
        key = (repr(color), specular, shininess)
        found = self._appearances.get(key)
        if found is None:
            try:
                found = Appearance(
                    color=(0.8, 0.8, 0.8) if color is None else color,
                    specular=0.0 if specular is None else specular,
                    shininess=32.0 if shininess is None else shininess,
                )
            except AetherError as exc:
                raise PlanError(
                    "INVALID_VALUE", getattr(exc, "message", str(exc)), pointer
                ) from None
            self._appearances[key] = found
        return found

    # -- names, ids and limits ------------------------------------------------------

    def _choose_names(self) -> None:
        taken = self.scene._by_name
        used: dict[str, int] = {}
        rename = self.plan.options.on_name_conflict == "rename"
        counters: dict[str, int] = {}
        renamed = 0
        objects = self.scene._objects
        for index, rec in enumerate(self.records):
            name = rec.new.name
            own = objects.get(rec.new.object_id) if rec.new.object_id is not None else None
            if own is not None and own.name == name:  # created by an earlier run of this plan
                used[name] = index
                continue
            if not name.strip():
                raise PlanError("INVALID_VALUE", "an object name is empty", rec.spec.pointer)
            if len(name) > 1024:
                raise PlanError(
                    "INVALID_VALUE", f"name {name[:40]!r}... is too long", rec.spec.pointer
                )
            clash = name in taken or name in used
            if clash and not rename:
                where = (
                    "in the scene"
                    if name in taken
                    else f"in this plan ({self.records[used[name]].path})"
                )
                raise PlanError(
                    "NAME_CONFLICT",
                    f"{rec.path}: the name {name!r} is already used {where}",
                    rec.spec.pointer,
                    hint='give a name template ("name": "Wall {i}"), another name, or set '
                    '"options": {"on_name_conflict": "rename"}',
                    details={"name": name, "path": rec.path},
                )
            if clash:
                k = counters.get(name, 1)
                while f"{name}.{k:03d}" in taken or f"{name}.{k:03d}" in used:
                    k += 1
                counters[name] = k + 1
                rec.new.name = f"{name}.{k:03d}"
                renamed += 1
            used[rec.new.name] = index
        if renamed:
            self.warn("NAMES_CHANGED", f"{renamed} name(s) were taken and got a .NNN suffix")

    def _check_ids(self) -> None:
        existing = self.scene._objects
        seen: set[str] = set()
        present = 0
        for rec in self.records:
            object_id = rec.new.object_id
            if object_id is None:
                continue
            if object_id in seen:
                raise PlanError(
                    "INVALID_VALUE", f"id {object_id!r} is used twice", rec.spec.pointer
                )
            seen.add(object_id)
            if object_id in existing:
                present += 1
        if present and self.plan.idempotency_key is None:
            raise PlanError(
                "NAME_CONFLICT",
                "an object with this id already exists in the scene",
                next(r.spec.pointer for r in self.records if r.new.object_id in existing),
            )
        self.present_ids = present
        chunked = (
            self.plan.options.chunk_size is not None
            and len(self.records) > self.plan.options.chunk_size
            and self.mode == "execute"
        )
        if present and not chunked:
            key = self.plan.idempotency_key
            if present == len(self.records) and self.mode == "execute":
                raise _AlreadyApplied(self)
            raise PlanError(
                "IDEMPOTENCY_CONFLICT",
                f"{present} of the {len(self.records)} objects of the plan with key {key!r} "
                "already exist: it was applied before, and the scene changed since",
                "/idempotency_key",
                hint="use a new idempotency key to apply the plan again",
                details={"present": present, "objects": len(self.records)},
            )

    def _check_limits(self) -> None:
        limits = self.limits
        total = len(self.scene) + len(self.records)
        if total > limits.max_scene_objects:
            raise PlanError(
                "LIMIT_EXCEEDED",
                f"the scene would hold {total} objects (limit {limits.max_scene_objects})",
                "/objects",
            )
        triangles = sum(r.new.mesh.triangle_count for r in self.records if r.new.mesh is not None)
        if triangles > limits.max_triangles:
            raise PlanError(
                "LIMIT_EXCEEDED",
                f"the new objects have {triangles} triangles (limit {limits.max_triangles})",
                "/objects",
                hint="share fewer, lighter meshes; procedural geometry is evaluated once per node",
            )
        new_bytes = sum(_mesh_bytes(m) for m in self._new_meshes())
        if new_bytes > limits.max_new_mesh_bytes:
            raise PlanError(
                "LIMIT_EXCEEDED",
                f"the new meshes take {new_bytes} bytes (limit {limits.max_new_mesh_bytes})",
                "/meshes",
            )
        self.triangles = triangles
        self.new_mesh_bytes = new_bytes

    def _new_meshes(self) -> list[Mesh]:
        """The meshes the new objects bring: every mesh this plan made. (The only meshes
        a plan takes from the scene are ``from_object`` ones, which are recorded.)"""
        out: dict[int, Mesh] = {}
        for rec in self.records:
            mesh = rec.new.mesh
            if mesh is not None and id(mesh) not in self._borrowed_meshes:
                out[id(mesh)] = mesh
        return list(out.values())

    def _chunks(self) -> list[range]:
        size = self.plan.options.chunk_size
        n = len(self.records)
        return [range(k, min(k + size, n)) for k in range(0, n, size)]

    # ------------------------------------------------------------------ #
    # Applying
    # ------------------------------------------------------------------ #

    def apply(self, indices: range, *, first: bool, last: bool) -> None:
        """Build one transaction's share of the plan (everything without chunking)."""
        plan, scene = self.plan, self.scene
        t = time.perf_counter()
        procedural = (
            plan.geometries
            or plan.procedural_parameters
            or plan.procedural_set
            or any(self.records[i].geometry is not None for i in indices)
        )
        if procedural:
            batch = scene.procedural.batch(f"{plan.label} (procedural)")
        else:
            batch = contextlib.nullcontext()
        with _Guard("/geometries" if plan.geometries else "/objects"), batch:
            if first:
                self._apply_procedural_parameters()
                self._apply_textures()
                self._apply_materials()
                self._apply_geometries()
                self.timings["resources"] = time.perf_counter() - t
                self.checkpoint(1, "resources")
            t = time.perf_counter()
            self._create_objects(indices)
            self.timings["objects"] = self.timings.get("objects", 0.0) + time.perf_counter() - t
        if last:
            t = time.perf_counter()
            self._activate_cameras()
            self._aim()
            self.checkpoint(1, "cameras and lights")
            self._apply_operations()
            self._apply_settings()
            self._frame_cameras()
            self.timings["operations"] = time.perf_counter() - t
            self.checkpoint(1, "operations")

    def _wrap(self, pointer: str, operation: int | None = None) -> _Guard:
        """Report an engine error raised inside the block as a ``PlanError`` at ``pointer``."""
        return _Guard(pointer, operation)

    def _apply_procedural_parameters(self) -> None:
        plan = self.plan
        if not (plan.procedural_parameters or plan.procedural_set):
            return
        system = self.scene.procedural
        for name, spec in plan.procedural_parameters.items():
            with self._wrap(f"/procedural/parameters/{name}"):
                options = {k: v for k, v in spec.items() if k != "value"}
                system.add_parameter(name, _proc_value(spec["value"]), **options)
        if plan.procedural_set:
            with self._wrap("/procedural/set"):
                system.set_parameters({k: _proc_value(v) for k, v in plan.procedural_set.items()})

    def _apply_textures(self) -> None:
        library = self.scene.materials
        for sym, spec in self.plan.textures.items():
            with self._wrap(spec.pointer):
                texture_id = self._resource_id("texture", sym)
                if spec.image is not None:
                    path = self._image_path(spec.image, f"{spec.pointer}/image")
                    options = dict(spec.options)
                    embed = bool(options.pop("embed", False))
                    texture = library.add_image_texture(
                        spec.name, path, embed=embed, texture_id=texture_id, **options
                    )
                else:
                    texture = library.add_procedural_texture(
                        spec.name, spec.graph, texture_id=texture_id
                    )
                self.textures[sym] = texture

    def _apply_materials(self) -> None:
        library = self.scene.materials
        for sym, spec in self.plan.materials.items():
            with self._wrap(spec.pointer):
                material_id = self._resource_id("material", sym)
                if spec.preset is not None:
                    material = library.create_material(
                        spec.name, preset=spec.preset, material_id=material_id, **spec.params
                    )
                else:
                    material = library.create_material(
                        spec.name, model=spec.model, material_id=material_id, **spec.params
                    )
                self.materials[sym] = material
            for slot, binding in spec.textures.items():
                with self._wrap(binding.pointer):
                    options = dict(binding.options)
                    sampler = {
                        k: options.pop(k)
                        for k in ("wrap", "filter", "mipmaps", "color_space")
                        if k in options
                    }
                    embed = bool(options.pop("embed", False))
                    if binding.texture is not None:
                        ref = binding.texture
                        texture = (
                            self.textures[ref.head]
                            if isinstance(ref, SymbolRef)
                            else library.texture(ref.value)
                        )
                    elif binding.image is not None:
                        texture = library.add_image_texture(
                            None,
                            self._image_path(binding.image, f"{binding.pointer}/image"),
                            embed=embed,
                            **sampler,
                        )
                    else:
                        texture = library.add_procedural_texture(
                            library._unique_texture_name(f"{material.name} {slot}"), binding.graph
                        )
                    material.set_texture(slot, texture, **options)

    def _resource_id(self, kind: str, symbol: str) -> str | None:
        key = self.plan.idempotency_key
        if key is None:
            return None
        return str(uuid.uuid5(ID_NAMESPACE, f"{key}\x1f{kind}:{symbol}"))

    def _image_path(self, value: str, pointer: str) -> Path:
        from ..materials.textures import check_relative_path

        if self.trusted_paths:
            path = Path(value)
            return path if path.is_absolute() else self.base_dir / path
        try:
            rel = check_relative_path(value, "image")
        except AetherError as exc:
            raise PlanError(
                "UNSAFE_PATH",
                getattr(exc, "message", str(exc)),
                pointer,
                hint="image paths are relative to the project's directory and stay inside it",
            ) from None
        path = (self.base_dir / rel).absolute()
        if not path.resolve().is_relative_to(self.base_dir.resolve()):
            raise PlanError("UNSAFE_PATH", f"image {value!r} leaves the project directory", pointer)
        return path

    def _apply_geometries(self) -> None:
        plan = self.plan
        if not plan.geometries:
            return
        scene = self.scene
        system = scene.procedural
        for sym in self._geometry_order():
            spec = plan.geometries[sym]
            with self._wrap(spec.pointer):
                if spec.primitive is not None:
                    node = system.add_geometry(
                        spec.name,
                        primitive=spec.primitive,
                        params={k: _proc_value(v) for k, v in spec.params.items()},
                    )
                elif spec.source is not None:
                    source = (
                        self.geometries[spec.source.head]
                        if isinstance(spec.source, SymbolRef)
                        else system.geometry(spec.source.value)
                    )
                    node = system.add_geometry(
                        spec.name,
                        source=source,
                        overrides={k: _proc_value(v) for k, v in spec.overrides.items()},
                        after=spec.after,
                    )
                else:
                    mesh = (
                        self.meshes[spec.mesh.head]
                        if isinstance(spec.mesh, SymbolRef)
                        else self._mesh_of(spec.mesh, f"{spec.pointer}/mesh")
                    )
                    node = system.add_geometry(spec.name, mesh=mesh)
                for pname, pspec in spec.parameters.items():
                    options = {k: v for k, v in pspec.items() if k != "value"}
                    system.add_parameter(
                        pname, _proc_value(pspec["value"]), geometry=node, **options
                    )
                system.evaluate(node)  # its source fails here, not at the end of the plan
            for m in spec.modifiers:
                with self._wrap(m.pointer):
                    params = {}
                    for k, v in m.params.items():
                        if (
                            k == "operand"
                            and isinstance(v, dict)
                            and isinstance(v.get("geometry"), (SymbolRef, SceneRef))
                        ):
                            ref = v["geometry"]
                            target = (
                                self.geometries[ref.head].id
                                if isinstance(ref, SymbolRef)
                                else system.geometry(ref.value).id
                            )
                            v = {**v, "geometry": target}
                        params[k] = _proc_value(v)
                    modifier = system.add_modifier(
                        node, m.type, name=m.name, enabled=m.enabled, params=params
                    )
                    system.evaluate(node, upto=modifier)
            self.geometries[sym] = node

    def _create_objects(self, indices: range) -> None:
        records = self.records
        scene = self.scene
        bulk = list(indices)
        for start in range(0, len(bulk), _SLICE):
            part = bulk[start : start + _SLICE]
            local = {g: k for k, g in enumerate(part)}
            news = []
            for g in part:
                rec = records[g]
                new = rec.new
                parent = rec.parent
                if isinstance(parent, int):
                    new.parent = local[parent] if parent in local else records[parent].created
                else:
                    new.parent = parent
                new.material = self._record_material(rec)
                new.group_materials = {
                    group: self.material_id(ref, f"{rec.spec.pointer}/group_materials/{group}")
                    for group, ref in rec.group_materials.items()
                    if ref is not None
                } or None
                news.append(new)
            with self._wrap(records[part[0]].spec.pointer):
                created = scene._create_many(news, self.plan.label)
            for g, obj in zip(part, created, strict=True):
                records[g].created = obj
                self.created_ids.add(obj._id)
            for g in part:
                rec = records[g]
                if rec.geometry is None:
                    continue
                node = rec.geometry
                node_id = self.geometries[node].id if node in self.geometries else node
                with self._wrap(f"{rec.spec.pointer}/geometry"):
                    scene.procedural.assign(rec.created, node_id)
            self.checkpoint(len(part), f"created {len(self.created_ids)} objects")

    def _record_material(self, rec: _Record) -> str | None:
        if rec.material is _BAD:
            return None
        return self.material_id(rec.material, f"{rec.spec.pointer}/material")

    def _activate_cameras(self) -> None:
        """Cameras become active as with ``add_camera``: when asked to, or if none is."""
        scene = self.scene
        for rec in self.records:
            if rec.camera is None or rec.created is None:
                continue
            active = rec.spec.active
            if active or (active is None and scene.active_camera is None):
                with self._wrap(rec.spec.pointer):
                    scene.active_camera = rec.created

    def _aim(self) -> None:
        for rec in self.records:
            if rec.aim is None:
                continue
            kind, target, up = rec.aim
            obj = rec.created
            pointer = rec.spec.pointer
            with self._wrap(pointer):
                upv = up if up is not None else (0.0, 0.0, 1.0)
                if kind == "at":
                    if isinstance(target, (SymbolRef, SceneRef)):
                        target = self.scene_object(target, f"{pointer}/look_at")
                    obj.look_at(target, up=upv)
                else:
                    obj.look_along(target, up=upv)

    def _frame_cameras(self) -> None:
        for rec in self.records:
            frame = rec.spec.frame
            if rec.camera is None or frame is None:
                continue
            pointer = f"{rec.spec.pointer}/frame"
            targets = None
            if frame.get("targets") is not None:
                targets = self.select(frame["targets"], pointer)
            with self._wrap(pointer):
                rec.created.frame(
                    targets,
                    direction=frame.get("direction"),
                    margin=frame.get("margin", 0.05),
                )

    def _apply_settings(self) -> None:
        plan, scene = self.plan, self.scene
        if plan.world is not None:
            with self._wrap("/world"):
                scene.world.configure(**plan.world)
        if plan.render_settings is not None:
            with self._wrap("/render_settings"):
                scene.render_settings.configure(**plan.render_settings)
        if plan.active_camera is not None:
            camera = self.scene_object(plan.active_camera, "/active_camera")
            with self._wrap("/active_camera"):
                scene.active_camera = camera

    # -- selections and operations ------------------------------------------------------

    def select(self, sel: Selector, pointer: str) -> list[SceneObject]:
        scene = self.scene
        if sel.explicit:
            found: dict[str, SceneObject] = {}
            for k, ref in enumerate(sel.refs):
                for obj in self.resolve_objects(ref, f"{pointer}/refs/{k}"):
                    found[obj._id] = obj
            for k, object_id in enumerate(sel.ids):
                obj = scene._objects.get(object_id)
                if obj is None:
                    raise PlanError(
                        "UNKNOWN_REFERENCE",
                        f"no object with id {object_id!r}",
                        f"{pointer}/ids/{k}",
                    )
                found[obj._id] = obj
            for k, name in enumerate(sel.names):
                obj = scene._by_name.get(name)
                if obj is None:
                    raise PlanError(
                        "UNKNOWN_REFERENCE", f"no object named {name!r}", f"{pointer}/names/{k}"
                    )
                found[obj._id] = obj
            objects = self._in_scene_order(list(found.values()))
        else:
            objects = list(scene)
        filters = sel.filters
        if filters:
            query: dict[str, Any] = {}
            for key in ("tags", "any_tags", "kind", "visible"):
                if key in filters:
                    query[key] = filters[key]
            if "name_glob" in filters:
                query["name"] = filters["name_glob"]
            if "under" in filters:
                query["ancestor"] = self.scene_object(filters["under"], f"{pointer}/under")
            if "material" in filters:
                query["material"] = self.material_id(filters["material"], f"{pointer}/material")
            with self._wrap(pointer):
                matching = {o._id for o in scene.find_objects(**query)}
            objects = [o for o in objects if o._id in matching]
        if sel.created:
            objects = [o for o in objects if o._id in self.created_ids]
        if sel.expect is not None:
            lo, hi = sel.expect
            if not lo <= len(objects) <= hi:
                raise PlanError(
                    "SELECTION_MISMATCH",
                    f"the selection matched {len(objects)} objects; expected "
                    + (f"{lo}" if lo == hi else f"{lo} to {hi}"),
                    pointer,
                    hint="check the selector (or the expect count) against the scene",
                    details={"matched": len(objects), "names": [o.name for o in objects[:10]]},
                )
        return objects

    def _in_scene_order(self, objects: list[SceneObject]) -> list[SceneObject]:
        """``objects`` in scene order (depth first), without walking the whole scene: each
        is keyed by its sibling positions from the root, and only the sibling lists on
        their paths are indexed (an edit of one door costs the same in any scene)."""
        if len(objects) <= 1:
            return objects
        positions: dict[int, dict[str, int]] = {}

        def position(obj: SceneObject) -> int:
            parent = obj._parent
            table = positions.get(id(parent))
            if table is None:
                siblings = self.scene._roots if parent is None else parent._children
                table = positions[id(parent)] = {o._id: k for k, o in enumerate(siblings)}
            return table[obj._id]

        def key(obj: SceneObject) -> tuple[int, ...]:
            path = []
            node: SceneObject | None = obj
            while node is not None:
                path.append(position(node))
                node = node._parent
            return tuple(reversed(path))

        return sorted(objects, key=key)

    def _apply_operations(self) -> None:
        for op in self.plan.operations:
            self.checkpoint()
            result = {"op": op.op, "index": op.index}
            if op.op == "update":
                objects = self.select(op.select, f"{op.pointer}/select")
                self._update(objects, op)
                result["matched"] = len(objects)
                if not objects:
                    self.warn(
                        "EMPTY_SELECTION", "update selected no objects", f"{op.pointer}/select"
                    )
            elif op.op == "delete":
                objects = self.select(op.select, f"{op.pointer}/select")
                removed = self._delete(objects, op)
                result.update(matched=len(objects), removed=removed)
            elif op.op == "set_parameters":
                values = {k: _proc_value(v) for k, v in op.values["values"].items()}
                geometry = op.values.get("geometry")
                with self._wrap(op.pointer, op.index):
                    node = None
                    if geometry is not None:
                        node = (
                            self.geometries[geometry.head]
                            if isinstance(geometry, SymbolRef)
                            else self.scene.procedural.geometry(geometry.value)
                        )
                    self.scene.procedural.set_parameters(values, geometry=node)
                result["parameters"] = sorted(values)
            elif op.op == "frame_camera":
                camera = (
                    self.scene_object(op.values["camera"], f"{op.pointer}/camera")
                    if op.values.get("camera") is not None
                    else self.scene.active_camera
                )
                if camera is None or camera.camera is None:
                    raise PlanError(
                        "NO_CAMERA", "there is no camera to frame", op.pointer, operation=op.index
                    )
                targets = None
                if op.values.get("targets") is not None:
                    targets = self.select(op.values["targets"], f"{op.pointer}/targets")
                with self._wrap(op.pointer, op.index):
                    framing = camera.frame(
                        targets,
                        direction=op.values.get("direction"),
                        margin=op.values.get("margin", 0.05),
                    )
                result["camera"] = {"id": camera.id, "name": camera.name}
                result["framing"] = framing.to_dict()
            elif op.op == "set_active_camera":
                camera = self.scene_object(op.values["camera"], f"{op.pointer}/camera")
                with self._wrap(op.pointer, op.index):
                    self.scene.active_camera = camera
                result["camera"] = {"id": camera.id, "name": camera.name}
            self.op_results.append(result)
            self.checkpoint(1)

    def _update(self, objects: list[SceneObject], op: OperationSpec) -> None:
        values = op.values
        pointer = f"{op.pointer}/set"
        if "name" in values and len(objects) > 1:
            raise PlanError(
                "INVALID_VALUE",
                f"name sets one object's name; {len(objects)} are selected",
                f"{pointer}/name",
                operation=op.index,
            )
        material = _BAD
        if "material" in values:
            material = self.material_id(values["material"], f"{pointer}/material")
        parent = _BAD
        if "parent" in values:
            parent = (
                None
                if values["parent"] is None
                else self.scene_object(values["parent"], f"{pointer}/parent")
            )
        look_target = None
        if isinstance(values.get("look_at"), (SymbolRef, SceneRef)):
            look_target = self.scene_object(values["look_at"], f"{pointer}/look_at")
        up = values.get("up", (0.0, 0.0, 1.0))
        library = self.scene.materials
        if material is not _BAD and objects:
            with self._wrap(f"{pointer}/material", op.index):
                library.assign(objects, material)
        for obj in objects:
            with self._wrap(pointer, op.index):
                if "name" in values:
                    obj.name = values["name"]
                if parent is not _BAD:
                    obj.set_parent(
                        parent, keep_world_transform=values.get("keep_world_transform", False)
                    )
                if "scale" in values:
                    obj.scale = values["scale"]
                if "scale_by" in values:
                    obj.scale = tuple(
                        a * b for a, b in zip(obj.scale, values["scale_by"], strict=True)
                    )
                if "rotation" in values:
                    obj.rotation = values["rotation"]
                if "rotation_euler" in values:
                    obj.rotation_euler = values["rotation_euler"]
                if "position" in values:
                    obj.position = values["position"]
                if "translate" in values:
                    obj.position = tuple(
                        a + b for a, b in zip(obj.position, values["translate"], strict=True)
                    )
                if "look_at" in values:
                    obj.look_at(
                        look_target if look_target is not None else values["look_at"], up=up
                    )
                if "look_along" in values:
                    obj.look_along(values["look_along"], up=up)
                appearance = {
                    k: values[k] for k in ("color", "specular", "shininess") if k in values
                }
                if appearance:
                    obj._check_no_material(next(iter(appearance)))
                    obj.appearance.configure(**appearance)
                if "visible" in values:
                    obj.visible = values["visible"]
                if "tags" in values:
                    obj.tags = values["tags"]
                if "add_tags" in values:
                    obj.add_tags(*values["add_tags"])
                if "remove_tags" in values:
                    obj.remove_tags(*values["remove_tags"])
                if "metadata" in values:
                    merged = obj.metadata
                    for key, value in values["metadata"].items():
                        if value is None:
                            merged.pop(key, None)
                        else:
                            merged[key] = value
                    obj.metadata = merged

    def _delete(self, objects: list[SceneObject], op: OperationSpec) -> int:
        recursive = bool(op.values.get("recursive", True))
        removed = 0
        for obj in objects:
            if obj._scene is None:  # removed with an ancestor already
                continue
            count = 1 + (sum(1 for _ in obj.iter_descendants()) if recursive else 0)
            with self._wrap(f"{op.pointer}/select", op.index):
                self.scene.remove(obj, recursive=recursive)
            removed += count
        return removed

    def run_queries(self) -> None:
        if not self.plan.queries:
            return
        from ..spatial.request import run_query

        for spec in self.plan.queries:
            request = self._substitute(spec.request, spec.pointer)
            try:
                answer = run_query(self.scene, with_selections(self.scene, request, spec.pointer))
            except AetherError as exc:
                answer = {"query": spec.request.get("query"), "error": _material_error(exc)}
            self.query_results.append({"index": spec.index, **answer})

    def _substitute(self, value: Any, pointer: str) -> Any:
        """Query arguments with plan references ("@towers") replaced by object ids."""
        from .model import parse_ref_text

        if isinstance(value, str) and value.startswith("@"):
            ref = parse_ref_text(value)
            if ref is None:
                return value
            objects = self.resolve_objects(ref, pointer)
            ids = [o.id for o in objects]
            return ids[0] if len(ids) == 1 else ids
        if isinstance(value, list):
            out = []
            for k, item in enumerate(value):
                sub = self._substitute(item, f"{pointer}/{k}")
                if isinstance(item, str) and item.startswith("@") and isinstance(sub, list):
                    out.extend(sub)
                else:
                    out.append(sub)
            return out
        if isinstance(value, dict):
            return {k: self._substitute(v, f"{pointer}/{k}") for k, v in value.items()}
        return value

    # -- chunked execution ------------------------------------------------------------

    def apply_chunked(self, chunks: list[range]) -> dict[str, Any]:
        """One transaction per chunk; completed chunks stay (or roll back) on failure."""
        scene = self.scene
        history = scene.history
        key = self.plan.idempotency_key
        checkpoint = f"plan:{key}:before"
        existing = scene._objects
        steps: list[dict[str, Any]] = []
        skipped = 0
        start_marked = False
        for number, chunk in enumerate(chunks):
            ids = [self.records[i].new.object_id for i in chunk]
            present = sum(1 for i in ids if i in existing)
            if present == len(ids):  # done by an earlier, interrupted run
                for i in chunk:
                    self.records[i].created = existing[self.records[i].new.object_id]
                    self.created_ids.add(self.records[i].new.object_id)
                skipped += 1
                continue
            if present:
                raise PlanError(
                    "IDEMPOTENCY_CONFLICT",
                    f"chunk {number + 1} of {len(chunks)} is partly in the scene",
                    "/idempotency_key",
                    hint="use a new idempotency key",
                )
            if not start_marked:
                history.checkpoint(checkpoint)
                start_marked = True
                if number > 0:
                    self._adopt_resources()  # created with chunk 1 by an earlier run
            label = f"{self.plan.label} (chunk {number + 1}/{len(chunks)})"
            try:
                with history.transaction(label) as tx:
                    self.apply(chunk, first=number == 0, last=number == len(chunks) - 1)
            except BaseException as exc:
                completed = len(steps)
                rolled_back = False
                if self.plan.options.on_chunk_failure == "rollback" and completed:
                    history.goto(checkpoint)
                    rolled_back = True
                if isinstance(exc, PlanError):
                    exc.details.update(
                        chunks={
                            "total": len(chunks),
                            "completed": completed,
                            "skipped": skipped,
                            "failed": number + 1,
                            "rolled_back": rolled_back,
                            "checkpoint": checkpoint if completed and not rolled_back else None,
                        }
                    )
                raise
            if tx.entry is not None:
                entry = tx.entry.to_dict()
                entry.pop("paths", None)
                steps.append(entry)
        return {
            "total": len(chunks),
            "completed": len(steps),
            "skipped": skipped,
            "checkpoint": checkpoint if start_marked else None,
            "steps": steps,
            "step": steps[-1] if steps else None,
        }

    def _adopt_resources(self) -> None:
        """Find the materials, textures and geometry nodes an earlier run created."""
        library = self.scene.materials
        for sym in self.plan.materials:
            self.materials[sym] = library.material(self._resource_id("material", sym))
        for sym in self.plan.textures:
            self.textures[sym] = library.texture(self._resource_id("texture", sym))
        if self.plan.geometries:
            system = self.scene.procedural
            for sym, spec in self.plan.geometries.items():
                self.geometries[sym] = system.geometry(spec.name)

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #

    def _counts(self) -> dict[str, Any]:
        records = self.records
        kinds: dict[str, int] = {}
        for rec in records:
            kind = (
                "camera"
                if rec.camera is not None
                else "light"
                if rec.light is not None
                else "procedural"
                if rec.geometry is not None
                else "mesh"
                if rec.new.mesh is not None
                else "empty"
            )
            kinds[kind] = kinds.get(kind, 0) + 1
        meshes = {id(r.new.mesh) for r in records if r.new.mesh is not None}
        return {
            "objects": len(records),
            "by_kind": kinds,
            "meshes": len(meshes),
            "materials": len(self.plan.materials),
            "textures": len(self.plan.textures),
            "geometries": len(self.plan.geometries),
            "operations": len(self.plan.operations),
            "mesh_objects_sharing_a_mesh": sum(
                1
                for r in records
                if r.new.mesh is not None and self._mesh_users.get(id(r.new.mesh), 0) > 1
            ),
        }

    def _symbols(self) -> dict[str, Any]:
        """Created objects by definition: "houses", "houses/walls" (all instances)."""
        mode = self.plan.options.result_ids
        out: dict[str, Any] = {}
        for group, paths in self.groups.items():
            recs = [self.records[i] for path in paths for i in self.by_path[path]]
            entry: dict[str, Any] = {"count": len(recs)}
            show = mode == "all" or (mode == "auto" and len(recs) <= AUTO_IDS)

            def ref(rec: _Record) -> dict[str, str]:
                obj = rec.created
                return {
                    "id": obj.id if obj is not None else rec.new.object_id,
                    "name": obj.name if obj is not None else rec.new.name,
                }

            if mode != "none":
                if show:
                    entry["objects"] = [ref(r) for r in recs]
                else:
                    entry["first"] = ref(recs[0])
                    entry["last"] = ref(recs[-1])
            out[group] = entry
        return out

    def _estimate(self) -> dict[str, Any]:
        records = self.records
        mesh_bytes = getattr(self, "new_mesh_bytes", 0)
        per_object = 1500  # bytes per object (SceneObject, transform, indexes, history)
        draw_groups = len({id(r.new.mesh) for r in records if r.new.mesh is not None})
        return {
            "objects": len(records),
            "scene_objects_after": len(self.scene) + len(records),
            "triangles": getattr(self, "triangles", 0),
            "new_mesh_bytes": mesh_bytes,
            "memory_bytes": len(records) * per_object + mesh_bytes,
            "meshes": draw_groups,
            "operations": len(self.plan.operations),
            "queries": len(self.plan.queries),
            "chunks": (
                math.ceil(len(records) / self.plan.options.chunk_size)
                if self.plan.options.chunk_size and len(records) > self.plan.options.chunk_size
                else 1
            ),
            "note": "procedural geometry is evaluated when the plan runs; its triangles are "
            "not counted here",
        }


class _Guard:
    """``with _Guard(pointer):`` -- an engine error in the block becomes a ``PlanError``."""

    __slots__ = ("pointer", "operation")

    def __init__(self, pointer: str, operation: int | None = None) -> None:
        self.pointer = pointer
        self.operation = operation

    def __enter__(self) -> _Guard:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or isinstance(exc, PlanError):
            return False
        if not isinstance(exc, (AetherError, TypeError, ValueError)):
            return False  # not a refusal of the engine: a bug, reported as it is
        code = getattr(exc, "code", None)
        raise PlanError(
            "OPERATION_FAILED",
            f"{type(exc).__name__}"
            + (f" [{code}]" if code else "")
            + f": {getattr(exc, 'message', None) or str(exc)}",
            self.pointer,
            operation=self.operation,
            hint=getattr(exc, "hint", None),
            details={"error": _material_error(exc)},
        ) from exc


def _uses_instance(*values: Any) -> bool:
    """Whether any of these parsed values is an expression over instance variables."""
    from .model import INSTANCE_VARIABLES

    for value in values:
        exprs = [value] if isinstance(value, Expr) else []
        if isinstance(value, tuple):
            exprs = [v for v in value if isinstance(v, Expr)]
        if any(e.names & INSTANCE_VARIABLES for e in exprs):
            return True
    return False


class _AlreadyApplied(Exception):
    def __init__(self, run: _Run) -> None:
        super().__init__("already applied")
        self.run = run


def _proc_value(value: Any) -> Any:
    """Procedural values pass through as data; ``{"expr": ...}`` stays an expression."""
    if isinstance(value, dict) and set(value) == {"expr"}:
        return Expr(value["expr"])
    if isinstance(value, Expr):
        return value
    if isinstance(value, list):
        return [_proc_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _proc_value(v) for k, v in value.items()}
    return value


def _vec3(value: Any, pointer: str) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise PlanError("INVALID_VALUE", f"expected (x, y, z), got {value!r}", pointer)
    out = tuple(float(c) for c in value)
    if not all(math.isfinite(c) for c in out):
        raise PlanError("INVALID_VALUE", f"expected finite numbers, got {value!r}", pointer)
    return out  # type: ignore[return-value]


def _scale3(value: Any, pointer: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (float(value),) * 3
    return _vec3(value, pointer)


def _mesh_bytes(mesh: Mesh) -> int:
    total = 0
    for name in ("positions", "normals", "uvs", "triangles", "groups"):
        array = getattr(mesh, name, None)
        if array is not None:
            total += int(array.nbytes)
    return total


def _primitive_mesh(
    primitive: str, params: dict[str, Any], pointer: str, limits: PlanLimits
) -> Mesh:
    from .. import primitives

    try:
        if primitive == "cube":
            return primitives.create_cube(params.get("size", 1.0))
        if primitive == "sphere":
            return primitives.create_sphere(
                params.get("radius", 1.0), params.get("segments", 32), params.get("rings", 16)
            )
        if primitive == "plane":
            return primitives.create_plane(params.get("size", 1.0))
        from ..procedural.primitives import PRIMITIVES

        values = dict(params)
        if primitive == "box":
            size = values.pop("size", None)
            if size is not None:
                values = {"width": size[0], "depth": size[1], "height": size[2]}
        polys = PRIMITIVES[primitive].generate(values, pointer)
        mesh, _ = polys.to_mesh()
        return mesh
    except AetherError as exc:
        raise PlanError(
            "INVALID_VALUE", getattr(exc, "message", None) or str(exc), pointer
        ) from None
