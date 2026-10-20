"""Parsing and static validation of scene plans (``aether3d.scene-plan``, version 1).

A plan is JSON data; nothing in it is executed as code. :func:`parse_plan` checks the
whole document -- structure, types, ranges, symbols, references, expressions, patterns,
limits -- and turns it into typed specs, or raises :class:`~aether3d.errors.PlanError`
listing every problem it found (``issues``), each with a JSON pointer into the plan.
Checks that need the scene (existing objects, names, materials) happen later, when the
plan is bound to a scene (:mod:`aether3d.plans.executor`), still before any change.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..errors import PlanError
from ..procedural.params import Expr, ProceduralError, evaluate_expression
from .limits import PlanLimits

FORMAT_ID = "aether3d.scene-plan"
VERSION = 1

SYMBOL_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}\Z")
"""Symbols (the keys of a plan's sections) are short ASCII identifiers."""
PARAMETER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
INSTANCE_VARIABLES = frozenset({"i", "n", "u", "x", "y", "z", "angle"})
"""Variables of per-instance expressions (see ``docs/SCENE_PLANS.md``)."""
RESERVED_NAMES = INSTANCE_VARIABLES | {
    "pi",
    "true",
    "false",
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
}

TOP_LEVEL_KEYS = (
    "format",
    "version",
    "label",
    "description",
    "expected_revision",
    "idempotency_key",
    "options",
    "parameters",
    "procedural",
    "meshes",
    "textures",
    "materials",
    "geometries",
    "prefabs",
    "objects",
    "operations",
    "world",
    "render_settings",
    "active_camera",
    "queries",
)

PATTERNS = ("grid", "array", "circle", "line", "scatter", "instances", "repeat")
OBJECT_KINDS = ("mesh", "geometry", "prefab", "camera", "light")
VARY_FIELDS = (
    "offset",
    "rotation_euler",
    "scale",
    "color",
    "specular",
    "shininess",
    "material",
    "mesh",
    "geometry",
    "visible",
    "tags",
)
GENERATORS = ("cycle", "choose", "uniform", "linear", "values", "expr")
MESH_PRIMITIVES = (
    "cube",
    "box",
    "sphere",
    "plane",
    "cylinder",
    "cone",
    "torus",
    "grid",
    "uv_sphere",
)
OPERATIONS = (
    "update",
    "delete",
    "set_parameters",
    "frame_camera",
    "set_active_camera",
    "cad",
    "animation",
    "world",
)
UPDATE_FIELDS = (
    "name",
    "position",
    "rotation_euler",
    "rotation",
    "scale",
    "translate",
    "scale_by",
    "color",
    "specular",
    "shininess",
    "material",
    "visible",
    "tags",
    "add_tags",
    "remove_tags",
    "metadata",
    "parent",
    "keep_world_transform",
    "look_at",
    "look_along",
    "up",
)
SELECTOR_KEYS = (
    "refs",
    "ids",
    "names",
    "tags",
    "any_tags",
    "name_glob",
    "kind",
    "under",
    "visible",
    "material",
    "created",
    "expect",
)
CAMERA_FIELDS = (
    "projection", "fov", "fov_axis", "ortho_scale", "near", "far", "aspect_ratio",
    "sensor", "focal_length", "shift_x", "shift_y", "fstop", "focus_distance",
    "aperture_blades", "aperture_rotation", "aperture_ratio",
)  # fmt: skip
LIGHT_FIELDS = (
    "type", "color", "intensity", "range", "falloff", "radius", "angle", "spot_size",
    "spot_blend", "shape", "size", "size_y", "spread", "units", "temperature", "use_temperature",
    "exposure", "shadow",
)  # fmt: skip
_BAD = object()  # a value that failed validation (its issue is recorded)


def escape(key: str) -> str:
    """A key escaped for a JSON pointer (RFC 6901)."""
    return key.replace("~", "~0").replace("/", "~1")


def join(pointer: str, key: str | int) -> str:
    return f"{pointer}/{escape(key) if isinstance(key, str) else key}"


def fingerprint(plan: Any) -> str:
    """SHA-256 of the plan's canonical JSON (sorted keys): the identity of a request."""
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Issues
# --------------------------------------------------------------------------- #


class Issues:
    """Problems found while checking a plan (the first ``limit`` are kept)."""

    def __init__(self, limit: int = 25) -> None:
        self.items: list[dict[str, Any]] = []
        self.limit = limit
        self.total = 0

    def add(
        self,
        code: str,
        message: str,
        pointer: str,
        *,
        hint: str | None = None,
        details: dict | None = None,
        operation: int | None = None,
    ) -> None:
        self.total += 1
        if len(self.items) < self.limit:
            self.items.append(
                {
                    "code": code,
                    "message": message,
                    "pointer": pointer or "",
                    "operation": operation,
                    "hint": hint,
                    "details": details or {},
                }
            )

    def __bool__(self) -> bool:
        return bool(self.items)

    def error(self) -> PlanError:
        first = self.items[0]
        details = dict(first["details"])
        if self.total > len(self.items):
            details["more_problems"] = self.total - len(self.items)
        return PlanError(
            first["code"],
            first["message"],
            first["pointer"],
            operation=first["operation"],
            hint=first["hint"],
            details=details,
            issues=self.items,
        )

    def raise_if_any(self) -> None:
        if self.items:
            raise self.error()


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SymbolRef:
    """``"@towers[2]/roof"``: objects (or resources) created by this plan."""

    text: str
    path: tuple[tuple[str, int | None], ...]  # (symbol, instance index or None)

    @property
    def head(self) -> str:
        return self.path[0][0]


@dataclass(frozen=True)
class SceneRef:
    """An existing entity of the scene: by id or name (``"Ground"``, ``{"id": ...}``)."""

    value: str
    by: str  # "any" | "id" | "name"

    @property
    def text(self) -> str:
        return self.value if self.by == "any" else f"{self.by}:{self.value}"


Ref = SymbolRef | SceneRef

_SEGMENT_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_.-]{0,63})(?:\[(-?\d{1,9})\])?\Z")


def parse_ref_text(text: str) -> SymbolRef | None:
    """``"@a[2]/b"`` -> SymbolRef, or ``None`` if the text is not a valid symbol path."""
    if not text.startswith("@") or len(text) > 512:
        return None
    path = []
    for segment in text[1:].split("/"):
        m = _SEGMENT_RE.match(segment)
        if m is None:
            return None
        path.append((m.group(1), None if m.group(2) is None else int(m.group(2))))
    return SymbolRef(text, tuple(path))


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #


@dataclass
class MeshSpec:
    symbol: str
    pointer: str
    primitive: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] | None = None
    from_object: Ref | None = None

    def key(self) -> str | None:
        """Identical specs share one mesh (``None``: never shared by content)."""
        if self.primitive is None:
            return None
        return json.dumps([self.primitive, _jsonable(self.params)], sort_keys=True)


@dataclass
class TextureSpec:
    symbol: str
    pointer: str
    name: str
    image: str | None = None
    graph: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class BindingSpec:
    pointer: str
    texture: Ref | None = None  # an existing texture or a plan texture symbol
    image: str | None = None
    graph: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialSpec:
    symbol: str
    pointer: str
    name: str
    preset: str | None = None
    model: str = "standard"
    params: dict[str, Any] = field(default_factory=dict)
    textures: dict[str, BindingSpec] = field(default_factory=dict)


@dataclass
class ModifierSpec:
    pointer: str
    type: str
    name: str | None
    enabled: bool
    params: dict[str, Any]


@dataclass
class GeometrySpec:
    symbol: str
    pointer: str
    name: str
    primitive: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    source: Ref | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    after: str | int | None = None
    mesh: Ref | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    modifiers: list[ModifierSpec] = field(default_factory=list)

    def references(self) -> list[Ref]:
        refs: list[Ref] = []
        if self.source is not None:
            refs.append(self.source)
        for m in self.modifiers:
            operand = m.params.get("operand")
            if isinstance(operand, dict) and isinstance(operand.get("geometry"), Ref):
                refs.append(operand["geometry"])
        return refs


@dataclass
class Generator:
    pointer: str
    kind: str  # cycle | choose | uniform | linear | values | expr
    values: list[Any] = field(default_factory=list)
    low: Any = None
    high: Any = None
    weights: list[float] | None = None
    seed: int = 0
    expr: Expr | None = None


@dataclass
class Pattern:
    kind: str
    pointer: str
    count: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class NameTemplate:
    """A name with placeholders: ``"Tower {i}"``, ``"{parent} roof"``, ``"Brick {i:3}"``."""

    parts: tuple[str | tuple[str, int], ...]  # literal text or (placeholder, zero padding)
    text: str

    @property
    def placeholders(self) -> set[str]:
        return {p[0] for p in self.parts if isinstance(p, tuple)}


@dataclass
class ObjectSpec:
    symbol: str  # the key in its section
    path: str  # "towers/roof" (prefab content: "prefab:house/roof")
    pointer: str
    kind: str = "empty"  # empty | mesh | geometry | prefab | camera | light
    name: NameTemplate | None = None
    object_id: str | None = None
    mesh: Ref | MeshSpec | None = None
    geometry: Ref | None = None
    prefab: Ref | None = None
    camera: dict[str, Any] | None = None
    light: dict[str, Any] | None = None
    parent: Ref | None = None
    position: Any = None
    rotation_euler: Any = None
    rotation: tuple[float, float, float, float] | None = None
    scale: Any = None
    look_at: Any = None  # point (vec3) or Ref
    look_along: Any = None
    direction: Any = None
    up: Any = None
    material: Ref | None | object = _BAD  # _BAD: not given
    group_materials: dict[int, Ref | None] = field(default_factory=dict)
    color: Any = None
    specular: Any = None
    shininess: Any = None
    visible: bool = True
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    active: bool | None = None
    frame: dict[str, Any] | None = None
    pattern: Pattern | None = None
    where: Expr | None = None  # instances for which it is false are not created
    vary: dict[str, Generator] = field(default_factory=dict)
    children: list[ObjectSpec] = field(default_factory=list)

    @property
    def material_given(self) -> bool:
        return self.material is not _BAD


@dataclass
class PrefabSpec:
    symbol: str
    pointer: str
    content: ObjectSpec  # what each instance contains (its root's own content and children)


@dataclass
class Selector:
    pointer: str
    refs: list[Ref] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    explicit: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    created: bool = False
    expect: tuple[int, int] | None = None


@dataclass
class OperationSpec:
    index: int
    pointer: str
    op: str
    select: Selector | None = None
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuerySpec:
    index: int
    pointer: str
    request: dict[str, Any]


@dataclass
class Options:
    on_name_conflict: str = "error"
    tags: tuple[str, ...] = ()
    result_ids: str = "auto"
    chunk_size: int | None = None
    on_chunk_failure: str = "keep"
    timeout: float | None = None


@dataclass
class Plan:
    raw: dict[str, Any]
    fingerprint: str
    label: str
    expected_revision: int | None
    idempotency_key: str | None
    options: Options
    parameters: dict[str, Any]  # evaluated plan constants
    procedural_parameters: dict[str, dict[str, Any]]
    procedural_set: dict[str, Any]
    meshes: dict[str, MeshSpec]
    textures: dict[str, TextureSpec]
    materials: dict[str, MaterialSpec]
    geometries: dict[str, GeometrySpec]
    prefabs: dict[str, PrefabSpec]
    objects: list[ObjectSpec]
    operations: list[OperationSpec]
    world: dict[str, Any] | None
    render_settings: dict[str, Any] | None
    active_camera: Ref | None
    queries: list[QuerySpec]
    symbols: dict[str, str]  # symbol -> section ("meshes", "objects", ...)


# --------------------------------------------------------------------------- #
# The reader
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    if isinstance(value, Expr):
        return {"expr": value.source}
    if isinstance(value, (SymbolRef, SceneRef)):
        return value.text
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Reader:
    """Typed accessors that record problems (with JSON pointers) instead of raising."""

    def __init__(self, issues: Issues, limits: PlanLimits) -> None:
        self.issues = issues
        self.limits = limits
        self.parameters: dict[str, Any] = {}

    # -- structure ---------------------------------------------------------

    def object(self, value: Any, ptr: str, what: str = "an object") -> dict | None:
        if not isinstance(value, dict):
            self.issues.add("INVALID_TYPE", f"expected {what}, got {_kind(value)}", ptr)
            return None
        return value

    def keys(
        self, obj: dict, ptr: str, allowed: tuple[str, ...] | set[str], required: tuple = ()
    ) -> bool:
        ok = True
        for key in obj:
            if key not in allowed:
                close = _closest(key, allowed)
                self.issues.add(
                    "UNKNOWN_KEY",
                    f"unknown key {key!r}",
                    join(ptr, key),
                    hint=f"did you mean {close!r}?" if close else f"allowed: {', '.join(allowed)}",
                )
                ok = False
        for key in required:
            if key not in obj:
                self.issues.add("MISSING_KEY", f"{key!r} is required", join(ptr, key))
                ok = False
        return ok

    def symbol(self, key: Any, ptr: str) -> bool:
        if not isinstance(key, str) or not SYMBOL_RE.match(key):
            self.issues.add(
                "INVALID_SYMBOL",
                f"invalid symbol {key!r}",
                ptr,
                hint="symbols are 1-64 characters: letters, digits, '_', '-', '.' "
                "(not starting with '-' or '.')",
            )
            return False
        return True

    # -- scalars -----------------------------------------------------------

    def string(self, value: Any, ptr: str, *, max_length: int = 256, empty: bool = False) -> Any:
        if not isinstance(value, str) or (not empty and not value.strip()):
            self.issues.add(
                "INVALID_VALUE", f"expected a non-empty string, got {_kind(value)}", ptr
            )
            return _BAD
        if len(value) > max_length:
            self.issues.add("INVALID_VALUE", f"string longer than {max_length} characters", ptr)
            return _BAD
        return value

    def boolean(self, value: Any, ptr: str) -> Any:
        if not isinstance(value, bool):
            self.issues.add("INVALID_VALUE", f"expected true or false, got {_kind(value)}", ptr)
            return _BAD
        return value

    def integer(self, value: Any, ptr: str, lo: int | None = None, hi: int | None = None) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                self.issues.add("INVALID_VALUE", f"expected an integer, got {_kind(value)}", ptr)
                return _BAD
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            self.issues.add(
                "OUT_OF_RANGE",
                f"{value} is outside [{'-inf' if lo is None else lo}, "
                f"{'inf' if hi is None else hi}]",
                ptr,
            )
            return _BAD
        return value

    def expression(self, value: Any, ptr: str, variables: frozenset[str]) -> Any:
        """``{"expr": "..."}`` -> :class:`Expr`, checked against the available names."""
        if not isinstance(value, dict) or set(value) != {"expr"}:
            self.issues.add("INVALID_VALUE", 'an expression is {"expr": "..."}', ptr)
            return _BAD
        try:
            expr = Expr(value["expr"])
        except ProceduralError as exc:
            self.issues.add("INVALID_EXPRESSION", exc.message, join(ptr, "expr"))
            return _BAD
        unknown = sorted(expr.names - variables - set(self.parameters))
        if unknown:
            known = sorted(set(self.parameters) | variables)
            self.issues.add(
                "UNKNOWN_VARIABLE",
                f"expression uses unknown name(s): {', '.join(unknown)}",
                join(ptr, "expr"),
                hint=f"available: {', '.join(known) or 'none (define plan parameters)'}",
            )
            return _BAD
        return expr

    def number(
        self,
        value: Any,
        ptr: str,
        *,
        lo: float | None = None,
        hi: float | None = None,
        variables: frozenset[str] | None = None,
        integer: bool = False,
    ) -> Any:
        """A finite number (or, with ``variables``, an expression over them)."""
        if isinstance(value, dict) and variables is not None:
            return self.expression(value, ptr, variables)
        if integer:
            return self.integer(value, ptr, None if lo is None else int(lo), hi)
        if not _is_number(value) or not math.isfinite(value):
            self.issues.add("INVALID_VALUE", f"expected a finite number, got {_kind(value)}", ptr)
            return _BAD
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            self.issues.add(
                "OUT_OF_RANGE",
                f"{value:g} is outside [{'-inf' if lo is None else f'{lo:g}'}, "
                f"{'inf' if hi is None else f'{hi:g}'}]",
                ptr,
            )
            return _BAD
        return float(value)

    def vec3(self, value: Any, ptr: str, variables: frozenset[str] | None = None) -> Any:
        """``[x, y, z]`` (components may be expressions) or one expression for all three."""
        if isinstance(value, dict) and variables is not None:
            return self.expression(value, ptr, variables)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            self.issues.add("INVALID_VALUE", f"expected [x, y, z], got {_kind(value)}", ptr)
            return _BAD
        items = [self.number(v, join(ptr, k), variables=variables) for k, v in enumerate(value)]
        return _BAD if any(v is _BAD for v in items) else tuple(items)

    def scale(self, value: Any, ptr: str, variables: frozenset[str] | None = None) -> Any:
        if _is_number(value) or (isinstance(value, dict) and "expr" in value):
            number = self.number(value, ptr, variables=variables)
            if number is _BAD:
                return _BAD
            return (number, number, number) if not isinstance(number, Expr) else number
        return self.vec3(value, ptr, variables)

    def color(self, value: Any, ptr: str) -> Any:
        from ..rendering.color import parse_color

        try:
            parse_color(value, ptr)
        except Exception as exc:  # RenderError
            self.issues.add("INVALID_VALUE", getattr(exc, "message", str(exc)), ptr)
            return _BAD
        return value

    def tags(self, value: Any, ptr: str) -> Any:
        if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
            self.issues.add("INVALID_VALUE", "tags must be an array of strings", ptr)
            return _BAD
        from ..scene import _check_tags

        try:
            return _check_tags(value)
        except Exception as exc:  # ValidationError
            self.issues.add("INVALID_VALUE", str(exc), ptr)
            return _BAD

    def metadata(self, value: Any, ptr: str) -> Any:
        from ..scene import _check_metadata

        try:
            _check_metadata(value)
        except Exception as exc:  # ValidationError
            self.issues.add("INVALID_VALUE", str(exc), ptr)
            return _BAD
        return value

    def ref(self, value: Any, ptr: str) -> Any:
        """``"@symbol..."``, an existing id or name, or ``{"id": ...}`` / ``{"name": ...}``."""
        if isinstance(value, str):
            if value.startswith("@"):
                ref = parse_ref_text(value)
                if ref is None:
                    self.issues.add(
                        "INVALID_REFERENCE",
                        f"{value!r} is not a valid reference",
                        ptr,
                        hint='use "@symbol", "@symbol[index]" or "@symbol/child"; '
                        'an existing object named "@..." is {"name": "@..."}',
                    )
                    return _BAD
                return ref
            if not value or len(value) > 256:
                self.issues.add(
                    "INVALID_REFERENCE", "an object id or name is 1-256 characters", ptr
                )
                return _BAD
            return SceneRef(value, "any")
        if isinstance(value, dict) and len(value) == 1:
            ((by, text),) = value.items()
            if by in ("id", "name") and isinstance(text, str) and 0 < len(text) <= 256:
                return SceneRef(text, by)
        self.issues.add(
            "INVALID_REFERENCE",
            f"expected a reference, got {_kind(value)}",
            ptr,
            hint='"@symbol", an existing id or name, {"id": ...} or {"name": ...}',
        )
        return _BAD

    def point_or_ref(self, value: Any, ptr: str, variables: frozenset[str]) -> Any:
        if isinstance(value, (str, dict)) and not (isinstance(value, dict) and "expr" in value):
            return self.ref(value, ptr)
        return self.vec3(value, ptr, variables)


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"{str(value).lower()} (a boolean)"
    if isinstance(value, (int, float)):
        return f"{value!r} (a number)"
    if isinstance(value, str):
        return f"{value[:40]!r} (a string)" if len(value) <= 40 else "a long string"
    if isinstance(value, list):
        return f"an array of {len(value)}"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def _closest(key: Any, allowed: tuple[str, ...] | set[str]) -> str | None:
    import difflib

    if not isinstance(key, str):
        return None
    found = difflib.get_close_matches(key, list(allowed), n=1, cutoff=0.75)
    return found[0] if found else None


# --------------------------------------------------------------------------- #
# Name templates
# --------------------------------------------------------------------------- #

_PLACEHOLDER_RE = re.compile(r"\{\{|\}\}|\{([A-Za-z_]+)(?::(\d{1,2}))?\}|[{}]")
NAME_PLACEHOLDERS = ("i", "n", "x", "y", "z", "parent", "symbol")


def parse_name(text: Any, ptr: str, reader: Reader) -> NameTemplate | None:
    """Parse a name template; placeholders are only replaced, never evaluated."""
    if reader.string(text, ptr) is _BAD:
        return None
    parts: list[str | tuple[str, int]] = []
    literal = []
    position = 0
    for m in _PLACEHOLDER_RE.finditer(text):
        literal.append(text[position : m.start()])
        position = m.end()
        token = m.group(0)
        if token in ("{{", "}}"):
            literal.append(token[0])
            continue
        name = m.group(1)
        if name is None or name not in NAME_PLACEHOLDERS:
            reader.issues.add(
                "INVALID_VALUE",
                f"unknown placeholder {token!r} in name {text!r}",
                ptr,
                hint="placeholders: {i}, {i:3} (zero-padded), {n}, {x}, {y}, {z}, {parent}, "
                "{symbol}; write {{ and }} for braces",
            )
            return None
        if literal:
            parts.append("".join(literal))
            literal = []
        parts.append((name, int(m.group(2) or 0)))
    literal.append(text[position:])
    if "".join(literal):
        parts.append("".join(literal))
    return NameTemplate(tuple(parts), text)


def render_name(template: NameTemplate, values: dict[str, Any]) -> str:
    out = []
    for part in template.parts:
        if isinstance(part, str):
            out.append(part)
            continue
        name, width = part
        value = values.get(name, "")
        if width and isinstance(value, int):
            out.append(f"{value:0{width}d}")
        else:
            out.append(str(value))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_plan(data: Any, limits: PlanLimits | None = None) -> Plan:
    """Check a plan document and return its typed form (raises :class:`PlanError`)."""
    limits = limits or PlanLimits()
    issues = Issues()
    reader = Reader(issues, limits)
    if isinstance(data, (str, bytes)):
        if len(data) > limits.max_plan_bytes:
            raise PlanError(
                "LIMIT_EXCEEDED", f"the plan is larger than {limits.max_plan_bytes} bytes", ""
            )
        try:
            data = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PlanError("INVALID_JSON", f"the plan is not valid JSON: {exc}", "") from None
    if not isinstance(data, dict):
        raise PlanError("INVALID_TYPE", f"a plan is a JSON object, got {_kind(data)}", "")
    # A private copy in plain JSON types: tuples become arrays, and later changes to the
    # caller's dict cannot reach the plan. Anything that is not JSON is refused here.
    try:
        text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PlanError("INVALID_JSON", f"the plan is not JSON data: {exc}", "") from None
    size = len(text.encode("utf-8"))
    data = json.loads(text)
    if size > limits.max_plan_bytes:
        raise PlanError(
            "LIMIT_EXCEEDED",
            f"the plan is {size} bytes as JSON; the limit is {limits.max_plan_bytes}",
            "",
            hint="use patterns, prefabs and shared meshes instead of listing every object",
        )
    reader.keys(data, "", TOP_LEVEL_KEYS, ("format", "version"))
    if data.get("format") != FORMAT_ID:
        issues.add("INVALID_FORMAT", f"format must be {FORMAT_ID!r}", "/format")
    version = data.get("version")
    if "version" in data and (isinstance(version, bool) or version != VERSION):
        issues.add(
            "UNSUPPORTED_VERSION",
            f"version {data.get('version')!r} is not supported (supported: {VERSION})",
            "/version",
        )
    issues.raise_if_any()  # nothing else makes sense without the right format

    label = data.get("label", "Scene plan")
    if reader.string(label, "/label", max_length=200) is _BAD:
        label = "Scene plan"
    if "description" in data:
        reader.string(data["description"], "/description", max_length=4000, empty=True)
    expected = None
    if data.get("expected_revision") is not None:
        expected = reader.integer(data["expected_revision"], "/expected_revision", 0)
        expected = None if expected is _BAD else expected
    key = None
    if data.get("idempotency_key") is not None:
        key = reader.string(data["idempotency_key"], "/idempotency_key", max_length=128)
        key = None if key is _BAD else key
    options = _parse_options(reader, data.get("options", {}), "/options")
    parameters = _parse_parameters(reader, data.get("parameters", {}), "/parameters")
    reader.parameters = parameters

    symbols: dict[str, str] = {}

    def section(name: str) -> dict:
        raw = data.get(name, {})
        obj = reader.object(raw, f"/{name}", "an object of symbol: definition")
        if obj is None:
            return {}
        out = {}
        for sym, spec in obj.items():
            ptr = join(f"/{name}", sym)
            if not reader.symbol(sym, ptr):
                continue
            if sym in symbols:
                issues.add(
                    "DUPLICATE_SYMBOL",
                    f"symbol {sym!r} is already defined in {symbols[sym]}",
                    ptr,
                    hint="symbols are unique across meshes, textures, materials, geometries, "
                    "prefabs and objects",
                )
                continue
            symbols[sym] = name
            out[sym] = spec
        return out

    raw_meshes = section("meshes")
    raw_textures = section("textures")
    raw_materials = section("materials")
    raw_geometries = section("geometries")
    raw_prefabs = section("prefabs")
    raw_objects = section("objects")

    meshes = {s: m for s, v in raw_meshes.items() if (m := _parse_mesh(reader, s, v)) is not None}
    textures = {
        s: t for s, v in raw_textures.items() if (t := _parse_texture(reader, s, v)) is not None
    }
    materials = {
        s: m for s, v in raw_materials.items() if (m := _parse_material(reader, s, v)) is not None
    }
    geometries = {
        s: g for s, v in raw_geometries.items() if (g := _parse_geometry(reader, s, v)) is not None
    }
    prefabs = {}
    for sym, raw in raw_prefabs.items():
        content = _parse_object(
            reader, sym, f"prefab:{sym}", join("/prefabs", sym), raw, prefab=True
        )
        if content is not None:
            prefabs[sym] = PrefabSpec(sym, join("/prefabs", sym), content)
    objects = []
    for sym, raw in raw_objects.items():
        spec = _parse_object(reader, sym, sym, join("/objects", sym), raw, top=True)
        if spec is not None:
            objects.append(spec)

    proc_params, proc_set = _parse_procedural(reader, data.get("procedural"), "/procedural")
    operations = _parse_operations(reader, data.get("operations", []), "/operations")
    world = _parse_settings(reader, data.get("world"), "/world")
    settings = _parse_settings(reader, data.get("render_settings"), "/render_settings")
    active = None
    if data.get("active_camera") is not None:
        active = reader.ref(data["active_camera"], "/active_camera")
        active = None if active is _BAD else active
    queries = _parse_queries(reader, data.get("queries", []), "/queries")

    plan = Plan(
        raw=data,
        fingerprint=fingerprint(data),
        label=label,
        expected_revision=expected,
        idempotency_key=key,
        options=options,
        parameters=parameters,
        procedural_parameters=proc_params,
        procedural_set=proc_set,
        meshes=meshes,
        textures=textures,
        materials=materials,
        geometries=geometries,
        prefabs=prefabs,
        objects=objects,
        operations=operations,
        world=world,
        render_settings=settings,
        active_camera=active,
        queries=queries,
        symbols=symbols,
    )
    _check_symbols(reader, plan)
    if options.chunk_size is not None and key is None:
        issues.add(
            "INVALID_VALUE",
            "chunked plans need an idempotency_key (it makes completed chunks resumable)",
            "/options/chunk_size",
            hint='add "idempotency_key": "<unique text>" to the plan',
        )
    issues.raise_if_any()
    return plan


# -- options and parameters -------------------------------------------------------


def _parse_options(r: Reader, raw: Any, ptr: str) -> Options:
    options = Options()
    obj = r.object(raw, ptr)
    if obj is None:
        return options
    r.keys(
        obj,
        ptr,
        ("on_name_conflict", "tags", "result_ids", "chunk_size", "on_chunk_failure", "timeout"),
    )
    if "on_name_conflict" in obj:
        if obj["on_name_conflict"] not in ("error", "rename"):
            r.issues.add("INVALID_VALUE", 'on_name_conflict is "error" or "rename"', ptr)
        else:
            options.on_name_conflict = obj["on_name_conflict"]
    if "tags" in obj:
        tags = r.tags(obj["tags"], join(ptr, "tags"))
        if tags is not _BAD:
            options.tags = tags
    if "result_ids" in obj:
        if obj["result_ids"] not in ("auto", "all", "none"):
            r.issues.add("INVALID_VALUE", 'result_ids is "auto", "all" or "none"', ptr)
        else:
            options.result_ids = obj["result_ids"]
    if obj.get("chunk_size") is not None:
        size = r.integer(obj["chunk_size"], join(ptr, "chunk_size"), 100, 1_000_000)
        options.chunk_size = None if size is _BAD else size
    if "on_chunk_failure" in obj:
        if obj["on_chunk_failure"] not in ("keep", "rollback"):
            r.issues.add("INVALID_VALUE", 'on_chunk_failure is "keep" or "rollback"', ptr)
        else:
            options.on_chunk_failure = obj["on_chunk_failure"]
    if obj.get("timeout") is not None:
        timeout = r.number(obj["timeout"], join(ptr, "timeout"), lo=0.001, hi=86400)
        options.timeout = None if timeout is _BAD else timeout
    return options


def _parse_parameters(r: Reader, raw: Any, ptr: str) -> dict[str, Any]:
    """Plan constants, evaluated in order (each may use the ones before it)."""
    obj = r.object(raw, ptr)
    values: dict[str, Any] = {}
    if obj is None:
        return values
    for name, value in obj.items():
        p = join(ptr, name)
        if not isinstance(name, str) or not PARAMETER_RE.match(name) or name in RESERVED_NAMES:
            r.issues.add(
                "INVALID_SYMBOL",
                f"invalid parameter name {name!r}",
                p,
                hint="letters, digits and '_', not starting with a digit; not a reserved name "
                f"({', '.join(sorted(RESERVED_NAMES))})",
            )
            continue
        r.parameters = values
        if isinstance(value, dict):
            expr = r.expression(value, p, frozenset())
            if expr is _BAD:
                continue
            try:
                result = evaluate_expression(expr, dict(values))
            except ProceduralError as exc:
                r.issues.add("EXPRESSION_ERROR", exc.message, join(p, "expr"))
                continue
        elif isinstance(value, list):
            result = r.vec3(value, p)
            if result is _BAD:
                continue
        elif isinstance(value, bool) or _is_number(value):
            if _is_number(value) and not math.isfinite(value):
                r.issues.add("INVALID_VALUE", "parameters must be finite", p)
                continue
            result = value
        else:
            r.issues.add(
                "INVALID_VALUE", "a parameter is a number, boolean, [x, y, z] or expression", p
            )
            continue
        values[name] = result
    r.parameters = values
    return values


def _parse_procedural(r: Reader, raw: Any, ptr: str) -> tuple[dict, dict]:
    if raw is None:
        return {}, {}
    obj = r.object(raw, ptr)
    if obj is None:
        return {}, {}
    r.keys(obj, ptr, ("parameters", "set"))
    params = _parameter_definitions(r, obj.get("parameters", {}), join(ptr, "parameters"))
    settings = r.object(obj.get("set", {}), join(ptr, "set")) or {}
    return params, dict(settings)


_PARAMETER_KEYS = ("value", "type", "min", "max", "choices", "description", "unit")


def _parameter_definitions(r: Reader, raw: Any, ptr: str) -> dict[str, dict[str, Any]]:
    """Procedural parameter definitions: ``{name: value}`` or ``{name: {"value": ...}}``."""
    out: dict[str, dict[str, Any]] = {}
    for name, spec in (r.object(raw, ptr) or {}).items():
        p = join(ptr, name)
        if isinstance(spec, dict) and "value" in spec:
            if r.keys(spec, p, _PARAMETER_KEYS):
                out[name] = dict(spec)
        else:
            out[name] = {"value": spec}
    return out


def _parse_settings(r: Reader, raw: Any, ptr: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    obj = r.object(raw, ptr)
    return None if obj is None else dict(obj)


# -- resources ----------------------------------------------------------------------


_PRIMITIVE_PARAMS: dict[str, dict[str, tuple]] = {
    # name: (kind, lo, hi): "num" positive float, "int" segments, "bool"
    "cube": {"size": ("num", 1e-9, 1e9)},
    "box": {
        "size": ("vec", 1e-9, 1e9),
        "width": ("num", 1e-9, 1e9),
        "depth": ("num", 1e-9, 1e9),
        "height": ("num", 1e-9, 1e9),
    },
    "sphere": {
        "radius": ("num", 1e-9, 1e9),
        "segments": ("int", 3, 1024),
        "rings": ("int", 2, 1024),
    },
    "plane": {"size": ("num", 1e-9, 1e9)},
    "cylinder": {
        "radius": ("num", 1e-9, 1e9),
        "depth": ("num", 1e-9, 1e9),
        "segments": ("int", 3, 1024),
        "caps": ("bool",),
        "smooth": ("bool",),
    },
    "cone": {
        "radius_bottom": ("num", 0.0, 1e9),
        "radius_top": ("num", 0.0, 1e9),
        "depth": ("num", 1e-9, 1e9),
        "segments": ("int", 3, 1024),
        "caps": ("bool",),
        "smooth": ("bool",),
    },
    "torus": {
        "major_radius": ("num", 1e-9, 1e9),
        "minor_radius": ("num", 1e-9, 1e9),
        "major_segments": ("int", 3, 1024),
        "minor_segments": ("int", 3, 1024),
    },
    "grid": {
        "x_size": ("num", 1e-9, 1e9),
        "y_size": ("num", 1e-9, 1e9),
        "x_segments": ("int", 1, 1024),
        "y_segments": ("int", 1, 1024),
    },
    "uv_sphere": {
        "radius": ("num", 1e-9, 1e9),
        "segments": ("int", 3, 1024),
        "rings": ("int", 2, 1024),
    },
}


def parse_mesh_spec(r: Reader, symbol: str, raw: Any, ptr: str) -> MeshSpec | None:
    obj = r.object(raw, ptr, "a mesh definition object")
    if obj is None:
        return None
    sources = [k for k in ("primitive", "data", "from_object") if k in obj]
    if len(sources) != 1:
        r.issues.add(
            "INVALID_VALUE",
            "a mesh has exactly one of primitive, data and from_object",
            ptr,
            hint='e.g. {"primitive": "cube", "size": 2}',
        )
        return None
    spec = MeshSpec(symbol, ptr)
    if "from_object" in obj:
        r.keys(obj, ptr, ("from_object",))
        ref = r.ref(obj["from_object"], join(ptr, "from_object"))
        if ref is _BAD:
            return None
        spec.from_object = ref
        return spec
    if "data" in obj:
        r.keys(obj, ptr, ("data",))
        data = r.object(obj["data"], join(ptr, "data"))
        if data is None or not r.keys(
            data,
            join(ptr, "data"),
            ("positions", "triangles", "normals", "uvs", "groups"),
            ("positions", "triangles"),
        ):
            return None
        for key in ("positions", "triangles", "normals", "uvs", "groups"):
            value = data.get(key)
            if value is None and key not in ("positions", "triangles"):
                continue
            nested = key != "groups"
            if not isinstance(value, list) or (
                nested and not all(isinstance(v, list) for v in value)
            ):
                r.issues.add(
                    "INVALID_VALUE",
                    f"{key} is an array of " + ("arrays" if nested else "integers"),
                    join(join(ptr, "data"), key),
                )
                return None
        vertices = len(data["positions"])
        if vertices > r.limits.max_inline_vertices:
            r.issues.add(
                "LIMIT_EXCEEDED",
                f"inline meshes have at most {r.limits.max_inline_vertices} vertices",
                join(ptr, "data/positions"),
            )
            return None
        spec.data = dict(data)
        return spec
    primitive = obj["primitive"]
    if primitive not in MESH_PRIMITIVES:
        r.issues.add(
            "INVALID_VALUE",
            f"unknown primitive {primitive!r}",
            join(ptr, "primitive"),
            hint=f"primitives: {', '.join(MESH_PRIMITIVES)}",
        )
        return None
    schema = _PRIMITIVE_PARAMS[primitive]
    r.keys(obj, ptr, ("primitive", *schema))
    variables = frozenset()
    params: dict[str, Any] = {}
    for name, value in obj.items():
        if name == "primitive" or name not in schema:
            continue
        p = join(ptr, name)
        kind = schema[name]
        if kind[0] == "num":
            v = r.number(value, p, lo=kind[1], hi=kind[2], variables=variables)
        elif kind[0] == "int":
            v = r.integer(value, p, kind[1], min(kind[2], r.limits.max_segments))
        elif kind[0] == "vec":
            v = r.vec3(value, p, variables)
            if (
                v is not _BAD
                and not isinstance(v, Expr)
                and any(not isinstance(c, Expr) and c <= 0 for c in v)
            ):
                r.issues.add("OUT_OF_RANGE", "box sizes must be positive", p)
                v = _BAD
        else:
            v = r.boolean(value, p)
        if v is not _BAD:
            params[name] = v
    if primitive == "box" and "size" in params and {"width", "depth", "height"} & set(params):
        r.issues.add("INVALID_VALUE", "give size or width/depth/height, not both", ptr)
        return None
    spec.primitive = primitive
    spec.params = params
    return spec


def _parse_mesh(r: Reader, symbol: str, raw: Any) -> MeshSpec | None:
    return parse_mesh_spec(r, symbol, raw, join("/meshes", symbol))


_SAMPLER = ("wrap", "filter", "mipmaps", "color_space")


def _parse_texture(r: Reader, symbol: str, raw: Any) -> TextureSpec | None:
    ptr = join("/textures", symbol)
    obj = r.object(raw, ptr, "a texture definition object")
    if obj is None:
        return None
    name = obj.get("name", symbol)
    if r.string(name, join(ptr, "name")) is _BAD:
        return None
    if ("image" in obj) == ("procedural" in obj):
        r.issues.add("INVALID_VALUE", "a texture has exactly one of image and procedural", ptr)
        return None
    if "image" in obj:
        r.keys(obj, ptr, ("name", "image", "embed", *_SAMPLER))
        image = r.string(obj["image"], join(ptr, "image"), max_length=1024)
        if image is _BAD:
            return None
        options = {k: obj[k] for k in ("embed", *_SAMPLER) if k in obj}
        if not _check_texture_options(r, options, ptr):
            return None
        return TextureSpec(symbol, ptr, name, image=image, options=options)
    r.keys(obj, ptr, ("name", "procedural"))
    if not isinstance(obj["procedural"], (dict, str)):
        r.issues.add("INVALID_VALUE", "procedural is a node graph or a node type", ptr)
        return None
    return TextureSpec(symbol, ptr, name, graph=obj["procedural"])


def _check_texture_options(r: Reader, options: dict[str, Any], ptr: str) -> bool:
    """Types of image options (the library checks the values when the texture is made)."""
    ok = True
    for key in ("embed", "mipmaps"):
        if key in options and not isinstance(options[key], bool):
            r.issues.add("INVALID_VALUE", f"{key} is true or false", join(ptr, key))
            ok = False
    for key, choices in (
        ("filter", ("linear", "nearest")),
        ("color_space", ("auto", "srgb", "linear")),
    ):
        if key in options and options[key] not in choices:
            r.issues.add("INVALID_VALUE", f"{key} is one of {', '.join(choices)}", join(ptr, key))
            ok = False
    return ok


def _parse_material(r: Reader, symbol: str, raw: Any) -> MaterialSpec | None:
    from ..materials.material import MODELS, PARAM_SPECS, SLOTS, TextureBinding
    from ..materials.presets import COMMON_OPTIONS, PRESETS

    ptr = join("/materials", symbol)
    obj = r.object(raw, ptr, "a material definition object")
    if obj is None:
        return None
    name = obj.get("name", symbol)
    if r.string(name, join(ptr, "name")) is _BAD:
        return None
    preset = obj.get("preset")
    allowed = {"name", "model", "preset", "textures", *PARAM_SPECS}
    if preset is not None:
        if not isinstance(preset, str) or preset not in PRESETS:
            r.issues.add(
                "INVALID_VALUE",
                f"unknown preset {preset!r}",
                join(ptr, "preset"),
                hint=f"presets: {', '.join(PRESETS)}",
            )
            return None
        allowed |= set(COMMON_OPTIONS) | set(PRESETS[preset][2])
    r.keys(obj, ptr, allowed)
    model = obj.get("model", "standard")
    if model not in MODELS:
        r.issues.add("INVALID_VALUE", f"model is one of {', '.join(MODELS)}", join(ptr, "model"))
        return None
    params = {k: v for k, v in obj.items() if k not in ("name", "model", "preset", "textures")}
    textures: dict[str, BindingSpec] = {}
    raw_textures = obj.get("textures", {})
    tex_ptr = join(ptr, "textures")
    for slot, binding in (r.object(raw_textures, tex_ptr) or {}).items():
        p = join(tex_ptr, slot)
        if slot not in SLOTS:
            r.issues.add(
                "INVALID_VALUE", f"unknown texture slot {slot!r}", p, hint=", ".join(SLOTS)
            )
            continue
        b = r.object(binding, p, "a texture binding object")
        if b is None:
            continue
        sources = [k for k in ("texture", "image", "procedural") if k in b]
        if len(sources) != 1:
            r.issues.add(
                "INVALID_VALUE", "a binding has exactly one of texture, image and procedural", p
            )
            continue
        r.keys(b, p, ("texture", "image", "procedural", "embed", *_SAMPLER, *TextureBinding.FIELDS))
        spec = BindingSpec(p)
        if "texture" in b:
            ref = r.ref(b["texture"], join(p, "texture"))
            if ref is _BAD:
                continue
            spec.texture = ref
        elif "image" in b:
            image = r.string(b["image"], join(p, "image"), max_length=1024)
            if image is _BAD:
                continue
            spec.image = image
        else:
            spec.graph = b["procedural"]
        spec.options = {k: v for k, v in b.items() if k not in ("texture", "image", "procedural")}
        if not _check_texture_options(r, spec.options, p):
            continue
        textures[slot] = spec
    return MaterialSpec(symbol, ptr, name, preset, model, params, textures)


def _parse_geometry(r: Reader, symbol: str, raw: Any) -> GeometrySpec | None:
    ptr = join("/geometries", symbol)
    obj = r.object(raw, ptr, "a geometry definition object")
    if obj is None:
        return None
    name = obj.get("name", symbol)
    if r.string(name, join(ptr, "name")) is _BAD:
        return None
    sources = [k for k in ("primitive", "source", "mesh") if k in obj]
    if len(sources) != 1:
        r.issues.add(
            "INVALID_VALUE",
            "a geometry has exactly one of primitive, source and mesh",
            ptr,
            hint='e.g. {"primitive": "cylinder", "params": {"radius": 2}}',
        )
        return None
    r.keys(
        obj,
        ptr,
        (
            "name",
            "primitive",
            "params",
            "source",
            "overrides",
            "after",
            "mesh",
            "parameters",
            "modifiers",
        ),
    )
    from ..procedural.modifiers import MODIFIER_TYPES
    from ..procedural.primitives import PRIMITIVES

    spec = GeometrySpec(symbol, ptr, name)
    if "primitive" in obj:
        primitive = obj["primitive"]
        if not isinstance(primitive, str) or primitive not in PRIMITIVES:
            close = _closest(primitive, set(PRIMITIVES))
            r.issues.add(
                "INVALID_VALUE",
                f"unknown procedural primitive {primitive!r}",
                join(ptr, "primitive"),
                hint=f"did you mean {close!r}?"
                if close
                else f"primitives: {', '.join(PRIMITIVES)}",
            )
            return None
        spec.primitive = primitive
        params = r.object(obj.get("params", {}), join(ptr, "params"))
        spec.params = dict(params or {})
    elif "source" in obj:
        ref = r.ref(obj["source"], join(ptr, "source"))
        if ref is _BAD:
            return None
        spec.source = ref
        spec.overrides = dict(r.object(obj.get("overrides", {}), join(ptr, "overrides")) or {})
        if "after" in obj:
            spec.after = obj["after"]
    else:
        ref = r.ref(obj["mesh"], join(ptr, "mesh"))
        if ref is _BAD:
            return None
        spec.mesh = ref
    spec.parameters = _parameter_definitions(r, obj.get("parameters", {}), join(ptr, "parameters"))
    raw_mods = obj.get("modifiers", [])
    mods_ptr = join(ptr, "modifiers")
    if not isinstance(raw_mods, list):
        r.issues.add("INVALID_TYPE", "modifiers is an array", mods_ptr)
        return None
    for k, m in enumerate(raw_mods):
        p = join(mods_ptr, k)
        mobj = r.object(m, p, "a modifier object")
        if mobj is None or not r.keys(mobj, p, ("type", "name", "enabled", "params"), ("type",)):
            continue
        mtype = mobj["type"]
        if not isinstance(mtype, str) or mtype not in MODIFIER_TYPES:
            close = _closest(mtype, set(MODIFIER_TYPES))
            r.issues.add(
                "INVALID_VALUE",
                f"unknown modifier type {mtype!r}",
                join(p, "type"),
                hint=f"did you mean {close!r}?"
                if close
                else f"modifiers: {', '.join(MODIFIER_TYPES)}",
            )
            continue
        if "name" in mobj and r.string(mobj["name"], join(p, "name")) is _BAD:
            continue
        params = dict(r.object(mobj.get("params", {}), join(p, "params")) or {})
        operand = params.get("operand")
        if isinstance(operand, dict) and isinstance(operand.get("geometry"), str):
            ref = r.ref(operand["geometry"], join(p, "params/operand/geometry"))
            if ref is not _BAD:
                params["operand"] = {**operand, "geometry": ref}
        enabled = mobj.get("enabled", True)
        if not isinstance(enabled, bool):
            r.issues.add("INVALID_VALUE", "enabled is true or false", join(p, "enabled"))
            continue
        spec.modifiers.append(ModifierSpec(p, mobj["type"], mobj.get("name"), enabled, params))
    return spec


# -- objects --------------------------------------------------------------------------


_OBJECT_KEYS = (
    "name",
    "id",
    "mesh",
    "geometry",
    "prefab",
    "camera",
    "light",
    "parent",
    "position",
    "rotation_euler",
    "rotation",
    "scale",
    "look_at",
    "look_along",
    "direction",
    "up",
    "material",
    "group_materials",
    "color",
    "specular",
    "shininess",
    "visible",
    "tags",
    "metadata",
    "active",
    "frame",
    "vary",
    "where",
    "children",
    *PATTERNS,
)


def _parse_object(
    r: Reader,
    symbol: str,
    path: str,
    ptr: str,
    raw: Any,
    *,
    top: bool = False,
    prefab: bool = False,
    depth: int = 0,
) -> ObjectSpec | None:
    obj = r.object(raw, ptr, "an object definition")
    if obj is None:
        return None
    if depth > r.limits.max_depth:
        r.issues.add("LIMIT_EXCEEDED", f"objects nest at most {r.limits.max_depth} deep", ptr)
        return None
    r.keys(obj, ptr, _OBJECT_KEYS)  # unknown keys are reported; the known ones still checked
    spec = ObjectSpec(symbol, path, ptr)
    kinds = [k for k in OBJECT_KINDS if k in obj]
    if len(kinds) > 1:
        r.issues.add(
            "INVALID_VALUE",
            f"an object has at most one of mesh, geometry, prefab, camera and light "
            f"(got {', '.join(kinds)})",
            ptr,
        )
        return None
    spec.kind = kinds[0] if kinds else "empty"
    patterns = [k for k in PATTERNS if k in obj]
    if len(patterns) > 1:
        r.issues.add(
            "INVALID_VALUE", f"an object has at most one pattern (got {', '.join(patterns)})", ptr
        )
        return None
    if prefab:
        for key in ("parent", "position", "rotation_euler", "rotation", "scale", *PATTERNS):
            if key in obj:
                r.issues.add(
                    "INVALID_VALUE",
                    f"a prefab's own {key!r} comes from each instance",
                    join(ptr, key),
                    hint="set position, rotation, scale and patterns where the prefab is used",
                )
        if spec.kind in ("camera", "light", "prefab"):
            r.issues.add(
                "INVALID_VALUE", "a prefab is a mesh, geometry or empty object with children", ptr
            )
            return None
    if "parent" in obj and not top:
        r.issues.add(
            "INVALID_VALUE",
            "only top-level objects take a parent; children belong to their parent",
            join(ptr, "parent"),
        )
    variables = INSTANCE_VARIABLES
    instance = bool(patterns)

    if "name" in obj:
        spec.name = parse_name(obj["name"], join(ptr, "name"), r)
    if "id" in obj:
        if patterns:
            r.issues.add("INVALID_VALUE", "id is for single objects, not patterns", join(ptr, "id"))
        else:
            object_id = r.string(obj["id"], join(ptr, "id"), max_length=128)
            spec.object_id = None if object_id is _BAD else object_id
    if spec.kind == "mesh":
        value = obj["mesh"]
        if isinstance(value, dict) and not ({"id", "name"} >= set(value) and len(value) == 1):
            mesh = parse_mesh_spec(r, f"{path}#mesh", value, join(ptr, "mesh"))
            spec.mesh = mesh
        else:
            ref = r.ref(value, join(ptr, "mesh"))
            spec.mesh = None if ref is _BAD else ref
    elif spec.kind in ("geometry", "prefab"):
        ref = r.ref(obj[spec.kind], join(ptr, spec.kind))
        setattr(spec, spec.kind, None if ref is _BAD else ref)
    elif spec.kind == "camera":
        cam = r.object(obj["camera"], join(ptr, "camera"))
        if cam is not None and r.keys(cam, join(ptr, "camera"), CAMERA_FIELDS):
            spec.camera = dict(cam)
        if patterns:
            r.issues.add("INVALID_VALUE", "cameras cannot use patterns", ptr)
    elif spec.kind == "light":
        light = r.object(obj["light"], join(ptr, "light"))
        if light is not None and r.keys(light, join(ptr, "light"), LIGHT_FIELDS):
            spec.light = dict(light)
    if spec.kind in ("camera", "light") and "children" in obj:
        r.issues.add(
            "INVALID_VALUE", "cameras and lights have no children in plans", join(ptr, "children")
        )
    if "parent" in obj and top:
        ref = r.ref(obj["parent"], join(ptr, "parent"))
        spec.parent = None if ref is _BAD else ref
    if "position" in obj:
        spec.position = r.vec3(obj["position"], join(ptr, "position"), variables)
    if "rotation_euler" in obj and "rotation" in obj:
        r.issues.add("INVALID_VALUE", "give rotation_euler or rotation, not both", ptr)
    if "rotation_euler" in obj:
        spec.rotation_euler = r.vec3(obj["rotation_euler"], join(ptr, "rotation_euler"), variables)
    if "rotation" in obj:
        value = obj["rotation"]
        p = join(ptr, "rotation")
        if (
            not isinstance(value, list)
            or len(value) != 4
            or not all(_is_number(v) and math.isfinite(v) for v in value)
            or sum(v * v for v in value) < 1e-18
        ):
            r.issues.add("INVALID_VALUE", "rotation is a non-zero quaternion [x, y, z, w]", p)
        else:
            spec.rotation = tuple(float(v) for v in value)
    if "scale" in obj:
        spec.scale = r.scale(obj["scale"], join(ptr, "scale"), variables)
    for key in ("look_at",):
        if key in obj:
            spec.look_at = r.point_or_ref(obj[key], join(ptr, key), variables)
    for key in ("look_along", "direction", "up"):
        if key in obj:
            setattr(spec, key, r.vec3(obj[key], join(ptr, key), variables))
    if "look_at" in obj and ("look_along" in obj or "direction" in obj):
        r.issues.add("INVALID_VALUE", "give look_at or look_along/direction, not both", ptr)
    if "material" in obj:
        value = obj["material"]
        if value is None:
            spec.material = None
        else:
            ref = r.ref(value, join(ptr, "material"))
            spec.material = _BAD if ref is _BAD else ref
    if "group_materials" in obj:
        groups = r.object(obj["group_materials"], join(ptr, "group_materials"))
        for key, value in (groups or {}).items():
            p = join(join(ptr, "group_materials"), key)
            if not (isinstance(key, str) and key.isdigit() and int(key) <= 65535):
                r.issues.add("INVALID_VALUE", "face groups are integers 0-65535", p)
                continue
            ref = None if value is None else r.ref(value, p)
            if ref is not _BAD:
                spec.group_materials[int(key)] = ref
    if "color" in obj:
        spec.color = r.color(obj["color"], join(ptr, "color"))
    if "specular" in obj:
        spec.specular = r.number(obj["specular"], join(ptr, "specular"), lo=0, hi=1)
    if "shininess" in obj:
        spec.shininess = r.number(obj["shininess"], join(ptr, "shininess"), lo=1, hi=10000)
    if "visible" in obj:
        visible = r.boolean(obj["visible"], join(ptr, "visible"))
        spec.visible = True if visible is _BAD else visible
    if "tags" in obj:
        tags = r.tags(obj["tags"], join(ptr, "tags"))
        spec.tags = () if tags is _BAD else tags
    if "metadata" in obj:
        meta = r.metadata(obj["metadata"], join(ptr, "metadata"))
        spec.metadata = None if meta is _BAD else meta
    if "active" in obj:
        if spec.kind != "camera":
            r.issues.add("INVALID_VALUE", "active is for cameras", join(ptr, "active"))
        active = r.boolean(obj["active"], join(ptr, "active"))
        spec.active = None if active is _BAD else active
    if "frame" in obj:
        if spec.kind != "camera":
            r.issues.add("INVALID_VALUE", "frame is for cameras", join(ptr, "frame"))
        spec.frame = _parse_frame(r, obj["frame"], join(ptr, "frame"))
    if patterns:
        spec.pattern = _parse_pattern(r, patterns[0], obj[patterns[0]], join(ptr, patterns[0]))
    if "where" in obj:
        if not instance:
            r.issues.add("INVALID_VALUE", "where needs a pattern (it selects instances)", ptr)
        else:
            expr = r.expression(obj["where"], join(ptr, "where"), INSTANCE_VARIABLES)
            spec.where = None if expr is _BAD else expr
    if "vary" in obj:
        if not instance:
            r.issues.add("INVALID_VALUE", "vary needs a pattern (it varies instances)", ptr)
        else:
            count = spec.pattern.count if spec.pattern is not None else 0
            spec.vary = _parse_vary(r, obj["vary"], join(ptr, "vary"), count)
    if "children" in obj:
        children = r.object(obj["children"], join(ptr, "children"))
        for key, child in (children or {}).items():
            p = join(join(ptr, "children"), key)
            if not r.symbol(key, p):
                continue
            parsed = _parse_object(r, key, f"{path}/{key}", p, child, depth=depth + 1)
            if parsed is not None:
                spec.children.append(parsed)
    return spec


def _parse_frame(r: Reader, raw: Any, ptr: str) -> dict[str, Any] | None:
    if raw is True:
        return {}
    obj = r.object(raw, ptr, "true or a framing object")
    if obj is None:
        return None
    r.keys(obj, ptr, ("targets", "direction", "margin"))
    out: dict[str, Any] = {}
    if "targets" in obj:
        out["targets"] = parse_selector(r, obj["targets"], join(ptr, "targets"))
    if "direction" in obj:
        out["direction"] = r.vec3(obj["direction"], join(ptr, "direction"))
    if "margin" in obj:
        out["margin"] = r.number(obj["margin"], join(ptr, "margin"), lo=0, hi=0.95)
    return out


def _parse_pattern(r: Reader, kind: str, raw: Any, ptr: str) -> Pattern | None:
    limit = r.limits.max_pattern_count

    def count_at(value: Any, p: str) -> int:
        if isinstance(value, dict):  # a plan-parameter expression
            expr = r.expression(value, p, frozenset())
            if expr is _BAD:
                return 0
            try:
                value = evaluate_expression(expr, dict(r.parameters))
            except ProceduralError as exc:
                r.issues.add("EXPRESSION_ERROR", exc.message, p)
                return 0
        n = r.integer(value, p, 1, limit)
        return 0 if n is _BAD else n

    if kind == "repeat":
        n = count_at(raw, ptr)
        return Pattern(kind, ptr, n) if n else None
    if kind == "instances":
        return _parse_instances(r, raw, ptr)
    obj = r.object(raw, ptr, f"a {kind} pattern object")
    if obj is None:
        return None
    params: dict[str, Any] = {}
    if kind == "grid":
        r.keys(obj, ptr, ("count", "spacing", "center"), ("count",))
        raw_count = obj.get("count")
        if not isinstance(raw_count, list) or len(raw_count) not in (2, 3):
            r.issues.add(
                "INVALID_VALUE", "grid count is [nx, ny] or [nx, ny, nz]", join(ptr, "count")
            )
            return None
        counts = [count_at(c, join(join(ptr, "count"), k)) for k, c in enumerate(raw_count)]
        if not all(counts):
            return None
        counts += [1] * (3 - len(counts))
        total = counts[0] * counts[1] * counts[2]
        if total > limit:
            r.issues.add(
                "LIMIT_EXCEEDED", f"grid of {total} > {limit} instances", join(ptr, "count")
            )
            return None
        spacing = obj.get("spacing", 1.0)
        if _is_number(spacing) or isinstance(spacing, dict):
            s = r.number(spacing, join(ptr, "spacing"), variables=frozenset())
            spacing_v = _BAD if s is _BAD else (s, s, s)
        else:
            if isinstance(spacing, list) and len(spacing) == 2:
                spacing = [*spacing, 1.0]  # [sx, sy]: a flat grid
            spacing_v = r.vec3(spacing, join(ptr, "spacing"), frozenset())
        if spacing_v is _BAD:
            return None
        center = obj.get("center", False)
        if r.boolean(center, join(ptr, "center")) is _BAD:
            return None
        params = {"counts": tuple(counts), "spacing": spacing_v, "center": center}
        return Pattern(kind, ptr, total, params)
    if kind == "array":
        r.keys(obj, ptr, ("count", "offset", "rotation_step", "scale_step"), ("count",))
        n = count_at(obj.get("count"), join(ptr, "count"))
        params["offset"] = r.vec3(obj.get("offset", [1, 0, 0]), join(ptr, "offset"), frozenset())
        if "rotation_step" in obj:
            params["rotation_step"] = r.vec3(
                obj["rotation_step"], join(ptr, "rotation_step"), frozenset()
            )
        if "scale_step" in obj:
            params["scale_step"] = r.scale(obj["scale_step"], join(ptr, "scale_step"), frozenset())
        if not n or any(v is _BAD for v in params.values()):
            return None
        return Pattern(kind, ptr, n, params)
    if kind == "circle":
        r.keys(
            obj,
            ptr,
            ("count", "radius", "start_angle", "end_angle", "orient", "z"),
            ("count", "radius"),
        )
        n = count_at(obj.get("count"), join(ptr, "count"))
        none = frozenset()
        params["radius"] = r.number(obj.get("radius"), join(ptr, "radius"), lo=0, variables=none)
        params["start_angle"] = r.number(
            obj.get("start_angle", 0.0), join(ptr, "start_angle"), variables=none
        )
        # Without end_angle the instances go all the way round, evenly spaced.
        params["end_angle"] = (
            r.number(obj["end_angle"], join(ptr, "end_angle"), variables=none)
            if "end_angle" in obj
            else None
        )
        params["z"] = r.number(obj.get("z", 0.0), join(ptr, "z"), variables=none)
        orient = obj.get("orient", "none")
        if orient not in ("none", "outward", "inward", "tangent"):
            r.issues.add(
                "INVALID_VALUE",
                'orient is "none", "outward", "inward" or "tangent"',
                join(ptr, "orient"),
            )
            return None
        params["orient"] = orient
        if not n or any(v is _BAD for v in params.values()):
            return None
        return Pattern(kind, ptr, n, params)
    if kind == "line":
        r.keys(obj, ptr, ("from", "to", "count"), ("from", "to", "count"))
        n = count_at(obj.get("count"), join(ptr, "count"))
        params["from"] = r.vec3(obj.get("from"), join(ptr, "from"), frozenset())
        params["to"] = r.vec3(obj.get("to"), join(ptr, "to"), frozenset())
        if not n or any(v is _BAD for v in params.values()):
            return None
        return Pattern(kind, ptr, n, params)
    if kind == "scatter":
        r.keys(
            obj,
            ptr,
            ("count", "seed", "box", "circle", "z", "min_distance", "allow_fewer", "max_attempts"),
            ("count",),
        )
        n = count_at(obj.get("count"), join(ptr, "count"))
        regions = [k for k in ("box", "circle") if k in obj]
        if len(regions) != 1:
            r.issues.add("INVALID_VALUE", "scatter needs exactly one region: box or circle", ptr)
            return None
        region_ptr = join(ptr, regions[0])
        region = r.object(obj[regions[0]], region_ptr)
        if region is None:
            return None
        if regions[0] == "box":
            r.keys(region, region_ptr, ("min", "max"), ("min", "max"))
            lo = _vec2(r, region.get("min"), join(region_ptr, "min"))
            hi = _vec2(r, region.get("max"), join(region_ptr, "max"))
            if lo is _BAD or hi is _BAD:
                return None
            if lo[0] > hi[0] or lo[1] > hi[1]:
                r.issues.add("INVALID_VALUE", "box min must not exceed max", region_ptr)
                return None
            params["region"] = ("box", lo, hi)
        else:
            r.keys(region, region_ptr, ("center", "radius"), ("radius",))
            center = _vec2(r, region.get("center", [0, 0]), join(region_ptr, "center"))
            radius = r.number(region.get("radius"), join(region_ptr, "radius"), lo=0)
            if center is _BAD or radius is _BAD:
                return None
            params["region"] = ("circle", center, radius)
        params["seed"] = r.integer(obj.get("seed", 0), join(ptr, "seed"), 0, 2**63 - 1)
        params["z"] = r.number(obj.get("z", 0.0), join(ptr, "z"), variables=frozenset())
        params["min_distance"] = r.number(
            obj.get("min_distance", 0.0), join(ptr, "min_distance"), lo=0
        )
        params["allow_fewer"] = r.boolean(obj.get("allow_fewer", False), join(ptr, "allow_fewer"))
        params["max_attempts"] = r.integer(
            obj.get("max_attempts", 30), join(ptr, "max_attempts"), 1, 1000
        )
        if not n or any(v is _BAD for v in params.values()):
            return None
        return Pattern(kind, ptr, n, params)
    raise AssertionError(kind)  # pragma: no cover


def _vec2(r: Reader, value: Any, ptr: str) -> Any:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_is_number(v) and math.isfinite(v) for v in value)
    ):
        r.issues.add("INVALID_VALUE", "expected [x, y]", ptr)
        return _BAD
    return (float(value[0]), float(value[1]))


_INSTANCE_KEYS = (
    "position",
    "rotation_euler",
    "scale",
    "name",
    "color",
    "material",
    "visible",
    "tags",
    "metadata",
)


def _parse_instances(r: Reader, raw: Any, ptr: str) -> Pattern | None:
    limit = r.limits.max_pattern_count
    if isinstance(
        raw, dict
    ):  # columns: {"positions": [...], "rotations_euler": [...], "scales": [...]}
        r.keys(raw, ptr, ("positions", "rotations_euler", "scales"), ("positions",))
        columns: dict[str, list] = {}
        n = None
        for key in ("positions", "rotations_euler", "scales"):
            if key not in raw:
                continue
            column = raw[key]
            p = join(ptr, key)
            if not isinstance(column, list) or not column:
                r.issues.add("INVALID_VALUE", f"{key} is a non-empty array", p)
                return None
            if len(column) > limit:
                r.issues.add("LIMIT_EXCEEDED", f"{len(column)} > {limit} instances", p)
                return None
            if n is not None and len(column) != n:
                r.issues.add("INVALID_VALUE", f"{key} has {len(column)} entries, positions {n}", p)
                return None
            n = len(column)
            if key == "scales":
                values = [r.scale(v, join(p, k)) for k, v in enumerate(column)]
            else:
                values = [r.vec3(v, join(p, k)) for k, v in enumerate(column)]
            if any(v is _BAD for v in values):
                return None
            columns[key] = values
        return Pattern("instances", ptr, n or 0, {"columns": columns})
    if not isinstance(raw, list) or not raw:
        r.issues.add("INVALID_VALUE", "instances is a non-empty array or an object of columns", ptr)
        return None
    if len(raw) > limit:
        r.issues.add("LIMIT_EXCEEDED", f"{len(raw)} > {limit} instances", ptr)
        return None
    rows = []
    for k, item in enumerate(raw):
        p = join(ptr, k)
        obj = r.object(item, p, "an instance object")
        if obj is None or not r.keys(obj, p, _INSTANCE_KEYS):
            return None
        row: dict[str, Any] = {}
        if "position" in obj:
            row["position"] = r.vec3(obj["position"], join(p, "position"))
        if "rotation_euler" in obj:
            row["rotation_euler"] = r.vec3(obj["rotation_euler"], join(p, "rotation_euler"))
        if "scale" in obj:
            row["scale"] = r.scale(obj["scale"], join(p, "scale"))
        if "name" in obj:
            row["name"] = parse_name(obj["name"], join(p, "name"), r)
        if "color" in obj:
            row["color"] = r.color(obj["color"], join(p, "color"))
        if "material" in obj:
            row["material"] = (
                None if obj["material"] is None else r.ref(obj["material"], join(p, "material"))
            )
        if "visible" in obj:
            row["visible"] = r.boolean(obj["visible"], join(p, "visible"))
        if "tags" in obj:
            row["tags"] = r.tags(obj["tags"], join(p, "tags"))
        if "metadata" in obj:
            row["metadata"] = r.metadata(obj["metadata"], join(p, "metadata"))
        if any(v is _BAD or v is None and key == "name" for key, v in row.items()):
            return None
        rows.append(row)
    return Pattern("instances", ptr, len(rows), {"rows": rows})


def _parse_vary(r: Reader, raw: Any, ptr: str, count: int) -> dict[str, Generator]:
    obj = r.object(raw, ptr)
    out: dict[str, Generator] = {}
    if obj is None:
        return out
    r.keys(obj, ptr, VARY_FIELDS)
    for fld, raw_gen in obj.items():
        if fld not in VARY_FIELDS:
            continue
        p = join(ptr, fld)
        gen = r.object(raw_gen, p, "a generator object")
        if gen is None:
            continue
        kinds = [k for k in GENERATORS if k in gen]
        if len(kinds) != 1:
            r.issues.add(
                "INVALID_VALUE",
                f"a generator has exactly one of {', '.join(GENERATORS)}",
                p,
                hint='e.g. {"cycle": ["#a03020", "#3060a0"]} or {"uniform": [0.8, 1.2], "seed": 1}',
            )
            continue
        kind = kinds[0]
        r.keys(gen, p, (kind, "seed", "weights"))
        g = Generator(p, kind)
        seed = r.integer(gen.get("seed", 0), join(p, "seed"), 0, 2**63 - 1)
        g.seed = 0 if seed is _BAD else seed
        values_ptr = join(p, kind)
        if kind in ("cycle", "choose", "values"):
            values = gen[kind]
            if not isinstance(values, list) or not values:
                r.issues.add("INVALID_VALUE", f"{kind} is a non-empty array", values_ptr)
                continue
            if kind == "values" and len(values) != count:
                r.issues.add(
                    "INVALID_VALUE",
                    f"values has {len(values)} entries for {count} instances",
                    values_ptr,
                )
                continue
            parsed = [_vary_value(r, fld, v, join(values_ptr, k)) for k, v in enumerate(values)]
            if any(v is _BAD for v in parsed):
                continue
            g.values = parsed
            if "weights" in gen:
                weights = gen["weights"]
                if (
                    kind != "choose"
                    or not isinstance(weights, list)
                    or len(weights) != len(values)
                    or not all(_is_number(w) and math.isfinite(w) and w >= 0 for w in weights)
                    or not sum(weights) > 0
                ):
                    r.issues.add(
                        "INVALID_VALUE",
                        "weights (choose only) are non-negative numbers, one per value",
                        join(p, "weights"),
                    )
                    continue
                g.weights = [float(w) for w in weights]
        elif kind in ("uniform", "linear"):
            if fld not in ("offset", "rotation_euler", "scale", "specular", "shininess"):
                r.issues.add(
                    "INVALID_VALUE", f"{kind} varies numbers and vectors, not {fld}", values_ptr
                )
                continue
            pair = gen[kind]
            if not isinstance(pair, list) or len(pair) != 2:
                r.issues.add("INVALID_VALUE", f"{kind} is [low, high]", values_ptr)
                continue
            lo = _vary_value(r, fld, pair[0], join(values_ptr, 0))
            hi = _vary_value(r, fld, pair[1], join(values_ptr, 1))
            if lo is _BAD or hi is _BAD:
                continue
            g.low, g.high = lo, hi
        else:  # expr
            if fld not in ("offset", "rotation_euler", "scale", "specular", "shininess", "visible"):
                r.issues.add(
                    "INVALID_VALUE",
                    f"expr varies numbers, vectors and visible, not {fld}",
                    values_ptr,
                )
                continue
            expr = r.expression({"expr": gen["expr"]}, p, INSTANCE_VARIABLES)
            if expr is _BAD:
                continue
            g.expr = expr
        out[fld] = g
    return out


def _vary_value(r: Reader, fld: str, value: Any, ptr: str) -> Any:
    if fld in ("offset", "rotation_euler"):
        return r.vec3(value, ptr)
    if fld == "scale":
        return r.scale(value, ptr)
    if fld == "color":
        return r.color(value, ptr)
    if fld == "specular":
        return r.number(value, ptr, lo=0, hi=1)
    if fld == "shininess":
        return r.number(value, ptr, lo=1, hi=10000)
    if fld in ("material", "mesh", "geometry"):
        if value is None and fld == "material":
            return None
        return r.ref(value, ptr)
    if fld == "visible":
        return r.boolean(value, ptr)
    if fld == "tags":
        return r.tags(value, ptr)
    raise AssertionError(fld)  # pragma: no cover


# -- selectors, operations and queries -----------------------------------------------------


def parse_selector(r: Reader, raw: Any, ptr: str) -> Selector | None:
    """A selection of objects: a reference, a list of them, or a selector object."""
    sel = Selector(ptr)
    if isinstance(raw, (str, list)) or (
        isinstance(raw, dict) and set(raw) <= {"id", "name"} and len(raw) == 1
    ):
        if isinstance(raw, list) and not raw:
            r.issues.add("INVALID_VALUE", "an empty list selects nothing", ptr)
            return None
        items = raw if isinstance(raw, list) else [raw]
        for k, item in enumerate(items):
            ref = r.ref(item, join(ptr, k) if isinstance(raw, list) else ptr)
            if ref is _BAD:
                return None
            sel.refs.append(ref)
        sel.explicit = True
        return sel
    obj = r.object(raw, ptr, "a selector (reference, list of references or selector object)")
    if obj is None or not r.keys(obj, ptr, SELECTOR_KEYS):
        return None
    for key in ("refs", "ids", "names"):
        if key not in obj:
            continue
        values = obj[key]
        p = join(ptr, key)
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            r.issues.add("INVALID_VALUE", f"{key} is an array of non-empty strings", p)
            return None
        if key == "refs":
            for k, v in enumerate(values):
                ref = r.ref(v, join(p, k))
                if ref is _BAD:
                    return None
                sel.refs.append(ref)
        else:
            getattr(sel, key).extend(values)
        sel.explicit = True
    for key in ("tags", "any_tags"):
        if key in obj:
            tags = r.tags(obj[key], join(ptr, key))
            if tags is _BAD:
                return None
            sel.filters[key] = list(tags)
    if "name_glob" in obj:
        if r.string(obj["name_glob"], join(ptr, "name_glob")) is _BAD:
            return None
        sel.filters["name_glob"] = obj["name_glob"]
    if "kind" in obj:
        kinds = ("mesh", "procedural", "camera", "light", "empty")
        if obj["kind"] not in kinds:
            r.issues.add("INVALID_VALUE", f"kind is one of {', '.join(kinds)}", join(ptr, "kind"))
            return None
        sel.filters["kind"] = obj["kind"]
    if "under" in obj:
        ref = r.ref(obj["under"], join(ptr, "under"))
        if ref is _BAD:
            return None
        sel.filters["under"] = ref
    if "visible" in obj:
        if r.boolean(obj["visible"], join(ptr, "visible")) is _BAD:
            return None
        sel.filters["visible"] = obj["visible"]
    if "material" in obj:
        ref = r.ref(obj["material"], join(ptr, "material"))
        if ref is _BAD:
            return None
        sel.filters["material"] = ref
    if "created" in obj:
        if r.boolean(obj["created"], join(ptr, "created")) is _BAD:
            return None
        sel.created = obj["created"]
    if "expect" in obj:
        expect = obj["expect"]
        p = join(ptr, "expect")
        if isinstance(expect, int) and not isinstance(expect, bool) and expect >= 0:
            sel.expect = (expect, expect)
        elif isinstance(expect, dict) and set(expect) <= {"min", "max"} and expect:
            lo = expect.get("min", 0)
            hi = expect.get("max", 2**63 - 1)
            if (
                not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (lo, hi))
                or lo > hi
            ):
                r.issues.add("INVALID_VALUE", "expect is {min, max} with 0 <= min <= max", p)
                return None
            sel.expect = (lo, hi)
        else:
            r.issues.add("INVALID_VALUE", "expect is a count or {min, max}", p)
            return None
    if not sel.explicit and not sel.filters and not sel.created:
        r.issues.add(
            "INVALID_VALUE",
            "an empty selector would select every object",
            ptr,
            hint="name objects (refs, ids, names) or filter them (tags, name_glob, kind, "
            "under, ...)",
        )
        return None
    return sel


_OPERATION_KEYS = {
    "update": (("select", "set"), ()),
    "delete": (("select",), ("recursive",)),
    "set_parameters": (("values",), ("geometry",)),
    "frame_camera": ((), ("camera", "targets", "direction", "margin")),
    "set_active_camera": (("camera",), ()),
    "cad": (("script",), ()),
    "animation": (("script",), ()),
    "world": (("script",), ()),
}


def _parse_operations(r: Reader, raw: Any, ptr: str) -> list[OperationSpec]:
    if not isinstance(raw, list):
        r.issues.add("INVALID_TYPE", "operations is an array", ptr)
        return []
    if len(raw) > r.limits.max_operations:
        r.issues.add("LIMIT_EXCEEDED", f"at most {r.limits.max_operations} operations", ptr)
        return []
    out = []
    for k, item in enumerate(raw):
        p = join(ptr, k)
        obj = r.object(item, p, 'an operation object ({"op": ...})')
        if obj is None:
            continue
        op = obj.get("op")
        if not isinstance(op, str) or op not in _OPERATION_KEYS:
            r.issues.add(
                "UNKNOWN_OPERATION",
                f"unknown operation {op!r}",
                join(p, "op"),
                hint=f"operations: {', '.join(OPERATIONS)}",
                operation=k,
            )
            continue
        required, optional = _OPERATION_KEYS[op]
        before = r.issues.total
        r.keys(obj, p, ("op", *required, *optional), required)
        spec = OperationSpec(k, p, op)
        if "select" in obj:
            spec.select = parse_selector(r, obj["select"], join(p, "select"))
        if op == "update":
            spec.values = _parse_update(r, obj.get("set"), join(p, "set"))
        elif op == "delete":
            if "recursive" in obj:
                r.boolean(obj["recursive"], join(p, "recursive"))
            spec.values = {"recursive": obj.get("recursive", True)}
            sel = spec.select
            if sel is not None and not sel.explicit and sel.expect is None:
                r.issues.add(
                    "INVALID_VALUE",
                    "deleting by filters needs expect (the number of objects you mean)",
                    join(p, "select"),
                    hint='add "expect": N, or list the objects in refs, ids or names',
                    operation=k,
                )
        elif op == "set_parameters":
            values = r.object(obj.get("values"), join(p, "values"))
            spec.values = {"values": dict(values or {})}
            if "geometry" in obj:
                ref = r.ref(obj["geometry"], join(p, "geometry"))
                spec.values["geometry"] = None if ref is _BAD else ref
        elif op == "frame_camera":
            if "camera" in obj:
                ref = r.ref(obj["camera"], join(p, "camera"))
                spec.values["camera"] = None if ref is _BAD else ref
            frame = _parse_frame(
                r, {k2: v for k2, v in obj.items() if k2 in ("targets", "direction", "margin")}, p
            )
            spec.values.update(frame or {})
        elif op == "set_active_camera":
            ref = r.ref(obj.get("camera"), join(p, "camera"))
            spec.values["camera"] = None if ref is _BAD else ref
        elif op == "cad":
            from ..cad.script import check_script
            from ..errors import CadError

            try:
                spec.values["script"] = check_script(obj.get("script"))
            except CadError as exc:
                where = (exc.location or "").removeprefix("script").strip(".")
                where = where.replace("[", ".").replace("]", "").replace(".", "/")
                r.issues.add(
                    "INVALID_VALUE",
                    f"CAD script: {exc.message}",
                    join(p, "script") + ("/" + where if where else ""),
                    hint="a CAD script (format aether3d.cad-script); see aether3d://cad/guide",
                    operation=k,
                )
        elif op == "animation":
            from ..animation.script import AnimationScriptError
            from ..animation.script import check_script as check_animation

            try:
                spec.values["script"] = check_animation(obj.get("script"))
            except AnimationScriptError as exc:
                where = (exc.location or "").removeprefix("script").strip(".")
                where = where.replace("[", ".").replace("]", "").replace(".", "/")
                r.issues.add(
                    "INVALID_VALUE",
                    f"animation script: {exc.message}",
                    join(p, "script") + ("/" + where if where else ""),
                    hint="an animation script (format aether3d.animation-script); see "
                    "aether3d://animation/guide",
                    operation=k,
                )
        elif op == "world":
            from ..errors import WorldError
            from ..worlds.script import check_script as check_world

            try:
                spec.values["script"] = check_world(obj.get("script"))
            except WorldError as exc:
                where = (exc.location or "").removeprefix("script").strip(".")
                where = where.replace("[", ".").replace("]", "").replace(".", "/")
                r.issues.add(
                    "INVALID_VALUE",
                    f"world script: {exc.message}",
                    join(p, "script") + ("/" + where if where else ""),
                    hint="a world script (format aether3d.world-script); see "
                    "aether3d://worlds/guide",
                    operation=k,
                )
        for issue in r.issues.items[before:]:
            if issue["operation"] is None:
                issue["operation"] = k
        out.append(spec)
    return out


def _parse_update(r: Reader, raw: Any, ptr: str) -> dict[str, Any]:
    obj = r.object(raw, ptr, "an object of fields to set")
    out: dict[str, Any] = {}
    if obj is None:
        return out
    r.keys(obj, ptr, UPDATE_FIELDS)
    if not obj:
        r.issues.add("INVALID_VALUE", "set is empty", ptr)
    for key, value in obj.items():
        p = join(ptr, key)
        if key == "name":
            v = r.string(value, p)
        elif key in ("position", "rotation_euler", "translate", "look_along", "up"):
            v = r.vec3(value, p)
        elif key == "rotation":
            if (
                not isinstance(value, list)
                or len(value) != 4
                or not all(_is_number(c) and math.isfinite(c) for c in value)
            ):
                r.issues.add("INVALID_VALUE", "rotation is a quaternion [x, y, z, w]", p)
                v = _BAD
            else:
                v = tuple(float(c) for c in value)
        elif key in ("scale", "scale_by"):
            v = r.scale(value, p)
        elif key == "color":
            v = r.color(value, p)
        elif key == "specular":
            v = r.number(value, p, lo=0, hi=1)
        elif key == "shininess":
            v = r.number(value, p, lo=1, hi=10000)
        elif key in ("visible", "keep_world_transform"):
            v = r.boolean(value, p)
        elif key in ("tags", "add_tags", "remove_tags"):
            v = r.tags(value, p)
        elif key == "metadata":
            m = r.object(value, p)
            v = _BAD if m is None else m
        elif key in ("material", "parent"):
            v = None if value is None else r.ref(value, p)
        elif key == "look_at":
            v = r.point_or_ref(value, p, frozenset())
        else:
            continue
        if v is not _BAD:
            out[key] = v
    if "position" in out and "translate" in out:
        r.issues.add("INVALID_VALUE", "give position or translate, not both", ptr)
    if "scale" in out and "scale_by" in out:
        r.issues.add("INVALID_VALUE", "give scale or scale_by, not both", ptr)
    if "rotation" in out and "rotation_euler" in out:
        r.issues.add("INVALID_VALUE", "give rotation or rotation_euler, not both", ptr)
    if "keep_world_transform" in out and "parent" not in out:
        r.issues.add("INVALID_VALUE", "keep_world_transform goes with parent", ptr)
    return out


def _parse_queries(r: Reader, raw: Any, ptr: str) -> list[QuerySpec]:
    if not isinstance(raw, list):
        r.issues.add("INVALID_TYPE", "queries is an array of query requests", ptr)
        return []
    if len(raw) > r.limits.max_queries:
        r.issues.add("LIMIT_EXCEEDED", f"at most {r.limits.max_queries} queries", ptr)
        return []
    from ..spatial.request import QUERIES

    out = []
    for k, item in enumerate(raw):
        p = join(ptr, k)
        obj = r.object(item, p, 'a query request ({"query": ...})')
        if obj is None:
            continue
        if not isinstance(obj.get("query"), str) or obj["query"] not in QUERIES:
            r.issues.add(
                "INVALID_VALUE",
                f"unknown query {obj.get('query')!r}",
                join(p, "query"),
                hint=f"queries: {', '.join(QUERIES)}",
            )
            continue
        out.append(QuerySpec(k, p, dict(obj)))
    return out


# -- symbol references ------------------------------------------------------------------


_SINGULAR = {
    "meshes": "mesh",
    "textures": "texture",
    "materials": "material",
    "geometries": "geometry",
    "prefabs": "prefab",
    "objects": "object",
}


def _check_symbols(r: Reader, plan: Plan) -> None:
    """References to plan symbols must name a symbol of the right section."""
    sections = plan.symbols

    def check(ref: Any, ptr: str, wanted: tuple[str, ...]) -> None:
        if not isinstance(ref, SymbolRef):
            return
        head = ref.head
        found = sections.get(head)
        if found is None:
            close = _closest(head, set(sections))
            r.issues.add(
                "UNKNOWN_REFERENCE",
                f"{ref.text!r}: no symbol {head!r} in this plan",
                ptr,
                hint=f"did you mean '@{close}'?"
                if close
                else 'define it, or name an existing object without "@"',
            )
        elif found not in wanted:
            r.issues.add(
                "INVALID_REFERENCE",
                f"{ref.text!r} names a {_SINGULAR[found]}, not a "
                f"{' or '.join(_SINGULAR[w] for w in wanted)}",
                ptr,
                hint="a prefab is a template: reach its copies through the objects that use "
                "it (@object/child), or select them by tag"
                if found == "prefabs" and "objects" in wanted
                else None,
            )
        elif found != "objects" and (len(ref.path) > 1 or ref.path[0][1] is not None):
            r.issues.add("INVALID_REFERENCE", f"{ref.text!r}: only objects have instances", ptr)

    for g in plan.geometries.values():
        if g.source is not None:
            check(g.source, join(g.pointer, "source"), ("geometries",))
        if g.mesh is not None:
            check(g.mesh, join(g.pointer, "mesh"), ("meshes",))
        for m in g.modifiers:
            operand = m.params.get("operand")
            if isinstance(operand, dict) and isinstance(operand.get("geometry"), Ref):
                check(
                    operand["geometry"], join(m.pointer, "params/operand/geometry"), ("geometries",)
                )
    for m in plan.materials.values():
        for b in m.textures.values():
            if b.texture is not None:
                check(b.texture, join(b.pointer, "texture"), ("textures",))
    for mesh in plan.meshes.values():
        if mesh.from_object is not None:
            check(mesh.from_object, join(mesh.pointer, "from_object"), ("objects",))

    def walk(spec: ObjectSpec) -> None:
        if isinstance(spec.mesh, Ref):
            check(spec.mesh, join(spec.pointer, "mesh"), ("meshes",))
        if spec.geometry is not None:
            check(spec.geometry, join(spec.pointer, "geometry"), ("geometries",))
        if spec.prefab is not None:
            check(spec.prefab, join(spec.pointer, "prefab"), ("prefabs",))
        if spec.parent is not None:
            check(spec.parent, join(spec.pointer, "parent"), ("objects",))
        if isinstance(spec.look_at, Ref):
            check(spec.look_at, join(spec.pointer, "look_at"), ("objects",))
        if spec.material_given and spec.material is not None:
            check(spec.material, join(spec.pointer, "material"), ("materials",))
        for group, ref in spec.group_materials.items():
            if ref is not None:
                check(ref, join(join(spec.pointer, "group_materials"), str(group)), ("materials",))
        for fld, gen in spec.vary.items():
            wanted = {"material": ("materials",), "mesh": ("meshes",), "geometry": ("geometries",)}
            if fld in wanted:
                for k, v in enumerate(gen.values):
                    if v is not None:
                        check(v, join(join(gen.pointer, gen.kind), k), wanted[fld])
        if spec.pattern is not None and spec.pattern.kind == "instances":
            for k, row in enumerate(spec.pattern.params.get("rows", [])):
                if row.get("material") is not None:
                    check(
                        row["material"],
                        join(join(spec.pattern.pointer, k), "material"),
                        ("materials",),
                    )
        if spec.frame is not None and spec.frame.get("targets") is not None:
            _check_selector(spec.frame["targets"], check)
        for child in spec.children:
            walk(child)

    for prefab in plan.prefabs.values():
        walk(prefab.content)
    for spec in plan.objects:
        walk(spec)
    for op in plan.operations:
        before = r.issues.total
        if op.select is not None:
            _check_selector(op.select, check)
        for key in ("material", "parent", "camera", "geometry"):
            value = op.values.get(key)
            if isinstance(value, Ref):
                wanted = {
                    "material": ("materials",),
                    "parent": ("objects",),
                    "camera": ("objects",),
                    "geometry": ("geometries",),
                }[key]
                check(
                    value,
                    join(op.pointer, key if key not in ("material", "parent") else f"set/{key}"),
                    wanted,
                )
        if isinstance(op.values.get("look_at"), Ref):
            check(op.values["look_at"], join(op.pointer, "set/look_at"), ("objects",))
        if op.values.get("targets") is not None:
            _check_selector(op.values["targets"], check)
        for issue in r.issues.items[before:]:
            if issue["operation"] is None:
                issue["operation"] = op.index
    if plan.active_camera is not None:
        check(plan.active_camera, "/active_camera", ("objects",))

    def query_refs(value: Any, ptr: str) -> None:  # "@..." strings in query arguments
        if isinstance(value, str):
            ref = parse_ref_text(value)
            if ref is not None:
                check(ref, ptr, ("objects",))
        elif isinstance(value, list):
            for k, item in enumerate(value):
                query_refs(item, join(ptr, k))
        elif isinstance(value, dict):
            for key, item in value.items():
                if key != "query":
                    query_refs(item, join(ptr, key))

    for q in plan.queries:
        query_refs(q.request, q.pointer)


def _check_selector(sel: Selector, check) -> None:
    for k, ref in enumerate(sel.refs):
        check(ref, join(join(sel.pointer, "refs"), k), ("objects",))
    for key in ("under",):
        if isinstance(sel.filters.get(key), Ref):
            check(sel.filters[key], join(sel.pointer, key), ("objects",))
    if isinstance(sel.filters.get("material"), Ref):
        check(sel.filters["material"], join(sel.pointer, "material"), ("materials",))


def evaluate(value: Any, env: dict[str, Any], pointer: str) -> Any:
    """Evaluate a parsed value: numbers and tuples as they are, expressions with ``env``."""
    if isinstance(value, Expr):
        try:
            return evaluate_expression(value, env)
        except ProceduralError as exc:
            raise PlanError("EXPRESSION_ERROR", exc.message, pointer) from None
    if isinstance(value, tuple) and any(isinstance(v, Expr) for v in value):
        return tuple(evaluate(v, env, f"{pointer}/{k}") for k, v in enumerate(value))
    return value
