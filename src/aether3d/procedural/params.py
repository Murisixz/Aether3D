"""Typed parameters and safe expressions for procedural definitions.

A parameter value is either a literal (number, bool, string, 3-vector) or an
:class:`Expr` -- a small arithmetic expression over other parameters, e.g.
``Expr("width / 2 - wall_thickness")``. Expressions are parsed with Python's
``ast`` module and evaluated by walking a strict whitelist of node types:
numbers, parameter names, ``+ - * / // % **``, unary minus, comparisons,
``and``/``or``/``not``, conditional ``a if c else b``, 3-tuples and a fixed set
of functions (``min max abs floor ceil round sqrt sin cos tan clamp int
float``). There is no attribute access, indexing, string evaluation or call
of arbitrary functions, so expressions from untrusted JSON are safe.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

from ..errors import AetherError

PARAMETER_TYPES = ("float", "int", "bool", "vec3", "enum", "string")
MAX_EXPRESSION_LENGTH = 1000
MAX_EXPRESSION_NODES = 200


class ProceduralError(AetherError, ValueError):
    """A procedural definition, parameter or evaluation is invalid; nothing changed.

    ``code`` is stable (e.g. ``"INVALID_PARAMETER"``, ``"DEPENDENCY_CYCLE"``,
    ``"MODIFIER_FAILED"``); ``location`` names the offending element (e.g.
    ``geometry[Tower].modifiers[Battlements].params.count``); ``hint`` suggests
    a fix; ``details`` carries extra structured data.
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str | None = None,
        *,
        hint: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        self.details = details or {}
        text = f"[{code}] " + (f"{location}: " if location else "") + message
        if hint:
            text += f". Hint: {hint}"
        super().__init__(text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "hint": self.hint,
            **({"details": self.details} if self.details else {}),
        }


@dataclass(frozen=True)
class Expr:
    """An expression over parameter names, e.g. ``Expr("floor(width / spacing)")``."""

    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ProceduralError("INVALID_EXPRESSION", "an expression must be a non-empty string")
        _parse(self.source)

    @property
    def names(self) -> frozenset[str]:
        return _parse(self.source)[1]

    def to_json(self) -> dict[str, str]:
        return {"expr": self.source}

    def __repr__(self) -> str:
        return f"Expr({self.source!r})"


_FUNCTIONS = {
    "min": min,
    "max": max,
    "abs": abs,
    "floor": lambda x: int(math.floor(x)),
    "ceil": lambda x: int(math.ceil(x)),
    "round": lambda x, n=0: round(x, int(n)) if n else int(round(x)),
    "sqrt": math.sqrt,
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
    "int": int,
    "float": float,
}
_CONSTANTS = {"pi": math.pi}
MAX_INT_BITS = 1024
"""Integers produced by expression arithmetic are limited to this size."""
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}
_cache: dict[str, tuple[ast.Expression, frozenset[str]]] = {}


def _parse(source: str) -> tuple[ast.Expression, frozenset[str]]:
    if source in _cache:
        return _cache[source]
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ProceduralError("INVALID_EXPRESSION", "expression is too long")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ProceduralError(
            "INVALID_EXPRESSION", f"invalid expression {source!r}: {exc.msg}"
        ) from None
    names: set[str] = set()
    for count, node in enumerate(ast.walk(tree), start=1):
        if count > MAX_EXPRESSION_NODES:
            raise ProceduralError("INVALID_EXPRESSION", "expression is too complex")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCTIONS
                or node.keywords
            ):
                raise ProceduralError(
                    "INVALID_EXPRESSION",
                    f"only these functions are allowed: {', '.join(sorted(_FUNCTIONS))}",
                )
        elif isinstance(node, ast.Name):
            if node.id not in _FUNCTIONS and node.id not in _CONSTANTS:
                names.add(node.id)
        elif not isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Compare,
                ast.BoolOp,
                ast.IfExp,
                ast.Constant,
                ast.Tuple,
                ast.Load,
                ast.operator,
                ast.unaryop,
                ast.cmpop,
                ast.boolop,
            ),
        ):
            raise ProceduralError(
                "INVALID_EXPRESSION",
                f"{type(node).__name__} is not allowed in expressions ({source!r})",
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ProceduralError(
                "INVALID_EXPRESSION", "only numeric and boolean constants are allowed"
            )
        if isinstance(node, ast.BinOp) and type(node.op) not in _BINOPS:
            raise ProceduralError(
                "INVALID_EXPRESSION", f"operator {type(node.op).__name__} is not allowed"
            )
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            raise ProceduralError("INVALID_EXPRESSION", "only unary + - not are allowed")
        if isinstance(node, ast.Compare) and any(type(op) not in _CMPOPS for op in node.ops):
            raise ProceduralError("INVALID_EXPRESSION", "comparison operator not allowed")
    result = (tree, frozenset(names))
    if len(_cache) < 4096:
        _cache[source] = result
    return result


def evaluate_expression(expr: Expr, env: dict[str, Any]) -> Any:
    """Evaluate ``expr`` with parameter values from ``env``."""
    tree, names = _parse(expr.source)
    missing = sorted(names - set(env))
    if missing:
        raise ProceduralError(
            "UNKNOWN_PARAMETER",
            f"expression {expr.source!r} uses undefined parameter(s): {', '.join(missing)}",
        )

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            return env[node.id]
        if isinstance(node, ast.Tuple):
            return tuple(ev(e) for e in node.elts)
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            return (
                -v
                if isinstance(node.op, ast.USub)
                else (+v if isinstance(node.op, ast.UAdd) else not v)
            )
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            # Arithmetic is on numbers only (no tuple repetition), with bounded
            # integer size, so an expression cannot exhaust memory or time.
            for operand in (left, right):
                if not isinstance(operand, (int, float)):
                    raise ProceduralError(
                        "EXPRESSION_ERROR",
                        f"arithmetic needs numbers, got {type(operand).__name__} "
                        f"in {expr.source!r}",
                    )
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ProceduralError("INVALID_EXPRESSION", "exponent too large")
            result = _BINOPS[type(node.op)](left, right)
            if isinstance(result, int) and result.bit_length() > MAX_INT_BITS:
                raise ProceduralError("EXPRESSION_ERROR", f"{expr.source!r}: number too large")
            return result
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators, strict=True):
                right = ev(comp)
                if not _CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            values = [ev(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, ast.Call):
            return _FUNCTIONS[node.func.id](*(ev(a) for a in node.args))
        raise ProceduralError("INVALID_EXPRESSION", f"unsupported syntax in {expr.source!r}")

    try:
        value = ev(tree)
    except ProceduralError:
        raise
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as exc:
        raise ProceduralError(
            "EXPRESSION_ERROR", f"evaluating {expr.source!r} failed: {exc}"
        ) from None
    if isinstance(value, float) and not math.isfinite(value):
        raise ProceduralError("EXPRESSION_ERROR", f"{expr.source!r} is not finite")
    return value


@dataclass
class Parameter:
    """A typed, validated parameter of a scene or a procedural geometry.

    ``value`` is a literal or an :class:`Expr` over other parameters. ``min``,
    ``max`` (numbers) and ``choices`` (enums) are enforced on the *resolved*
    value, so derived values are range-checked too.
    """

    name: str
    type: str
    value: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None
    description: str = ""
    unit: str = ""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "value": value_to_json(self.value),
        }
        if self.min is not None:
            data["min"] = self.min
        if self.max is not None:
            data["max"] = self.max
        if self.choices is not None:
            data["choices"] = list(self.choices)
        if self.description:
            data["description"] = self.description
        if self.unit:
            data["unit"] = self.unit
        return data

    def schema(self) -> dict[str, Any]:
        return {k: v for k, v in self.to_json().items() if k != "value"}


def value_to_json(value: Any) -> Any:
    if isinstance(value, Expr):
        return value.to_json()
    if isinstance(value, tuple):
        return [value_to_json(v) for v in value]
    if isinstance(value, dict):
        return {k: value_to_json(v) for k, v in value.items()}
    return value


def value_from_json(value: Any) -> Any:
    """Turn JSON into parameter values: ``{"expr": ...}`` becomes :class:`Expr`."""
    if isinstance(value, dict):
        if set(value) == {"expr"}:
            return Expr(value["expr"])
        return {k: value_from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(value_from_json(v) for v in value)
    return value


def is_identifier(name: Any) -> bool:
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not name.startswith("_")
        and name not in _FUNCTIONS
        and name not in _CONSTANTS
        and name not in ("and", "or", "not", "if", "else", "True", "False", "None")
    )


def check_type(ptype: str, value: Any, *, minimum=None, maximum=None, choices=None) -> Any:
    """Validate a resolved value against a parameter type and constraints; returns it normalized."""
    if ptype == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProceduralError("INVALID_PARAMETER", f"expected a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise ProceduralError("INVALID_PARAMETER", f"{value!r} is not finite")
    elif ptype == "int":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) != int(value)
        ):
            raise ProceduralError("INVALID_PARAMETER", f"expected an integer, got {value!r}")
        value = int(value)
    elif ptype == "bool":
        if not isinstance(value, bool):
            raise ProceduralError("INVALID_PARAMETER", f"expected true/false, got {value!r}")
    elif ptype == "vec3":
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ProceduralError("INVALID_PARAMETER", f"expected 3 numbers, got {value!r}")
        out = []
        for v in value:
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(float(v))
            ):
                raise ProceduralError(
                    "INVALID_PARAMETER", f"expected 3 finite numbers, got {value!r}"
                )
            out.append(float(v))
        value = tuple(out)
    elif ptype == "enum":
        if value not in (choices or ()):
            raise ProceduralError(
                "INVALID_PARAMETER", f"expected one of {list(choices or ())}, got {value!r}"
            )
    elif ptype == "string":
        if not isinstance(value, str):
            raise ProceduralError("INVALID_PARAMETER", f"expected a string, got {value!r}")
    else:
        raise ProceduralError("INVALID_PARAMETER", f"unknown parameter type {ptype!r}")
    if ptype in ("float", "int"):
        if minimum is not None and value < minimum:
            raise ProceduralError(
                "PARAMETER_OUT_OF_RANGE", f"{value!r} is below the minimum {minimum}"
            )
        if maximum is not None and value > maximum:
            raise ProceduralError(
                "PARAMETER_OUT_OF_RANGE", f"{value!r} is above the maximum {maximum}"
            )
    return value


def make_parameter(
    name: str,
    value: Any,
    *,
    type: str | None = None,
    min: float | None = None,
    max: float | None = None,
    choices: Any = None,
    description: str = "",
    unit: str = "",
) -> Parameter:
    """Create a :class:`Parameter`, inferring ``type`` from a literal value when omitted."""
    if not is_identifier(name):
        raise ProceduralError(
            "INVALID_PARAMETER", f"parameter name {name!r} must be a plain identifier"
        )
    value = value_from_json(value) if isinstance(value, (dict, list)) else value
    if type is None:
        if isinstance(value, Expr):
            raise ProceduralError(
                "INVALID_PARAMETER", "give type= for parameters defined by expressions"
            )
        type = (
            "bool"
            if isinstance(value, bool)
            else "int"
            if isinstance(value, int)
            else "float"
            if isinstance(value, float)
            else "vec3"
            if isinstance(value, tuple)
            else "enum"
            if choices is not None
            else "string"
        )
    if type not in PARAMETER_TYPES:
        raise ProceduralError(
            "INVALID_PARAMETER", f"type must be one of {PARAMETER_TYPES}, got {type!r}"
        )
    if choices is not None:
        choices = tuple(choices)
        if type != "enum" or not choices or not all(isinstance(c, str) for c in choices):
            raise ProceduralError(
                "INVALID_PARAMETER", "choices must be a non-empty list of strings (enum)"
            )
    elif type == "enum":
        raise ProceduralError("INVALID_PARAMETER", "enum parameters need choices")
    for bound in (min, max):
        if bound is not None and (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(bound)
        ):
            raise ProceduralError("INVALID_PARAMETER", "min/max must be finite numbers")
    if min is not None and max is not None and min > max:
        raise ProceduralError("INVALID_PARAMETER", f"min ({min}) is greater than max ({max})")
    if not isinstance(value, Expr):
        check_type(type, value, minimum=min, maximum=max, choices=choices)
    return Parameter(name, type, value, min, max, choices, description, unit)


@dataclass
class Rule:
    """A named condition over parameters that must hold (forbids invalid combinations).

    ``expr`` is a boolean :class:`Expr`; when it evaluates false, the change
    that caused it is rejected with ``CONSTRAINT_VIOLATED`` and ``message``.
    """

    name: str
    expr: Expr
    message: str = ""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "expr": self.expr.source}
        if self.message:
            data["message"] = self.message
        return data


def make_rule(name: str, expr: Any, message: str = "") -> Rule:
    if not is_identifier(name):
        raise ProceduralError("INVALID_RULE", f"rule name {name!r} must be a plain identifier")
    if isinstance(expr, str):
        expr = Expr(expr)
    if not isinstance(expr, Expr):
        raise ProceduralError("INVALID_RULE", "a rule needs an expression string")
    if not isinstance(message, str):
        raise ProceduralError("INVALID_RULE", "a rule message must be a string")
    return Rule(name, expr, message)


def check_rules(rules: dict[str, Rule], env: dict[str, Any], location: str) -> None:
    """Raise CONSTRAINT_VIOLATED for the first rule that does not hold."""
    for rule in rules.values():
        loc = f"{location}.rules[{rule.name}]"
        try:
            ok = evaluate_expression(rule.expr, env)
        except ProceduralError as exc:
            raise ProceduralError(exc.code, exc.message, loc) from None
        if not isinstance(ok, bool):
            raise ProceduralError(
                "INVALID_RULE", f"rule {rule.name!r} must evaluate to true or false", loc
            )
        if not ok:
            values = {n: value_to_json(env[n]) for n in sorted(rule.expr.names)}
            raise ProceduralError(
                "CONSTRAINT_VIOLATED",
                rule.message or f"rule {rule.name!r} ({rule.expr.source}) does not hold",
                loc,
                details={"rule": rule.to_json(), "values": values},
            )


@dataclass(frozen=True)
class ParamSpec:
    """Schema of one parameter of a primitive or modifier type."""

    name: str
    type: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None
    description: str = ""
    required: bool = False

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }
        if self.required:
            data["required"] = True
        else:
            data["default"] = value_to_json(self.default)
        for key in ("min", "max"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        if self.choices is not None:
            data["choices"] = list(self.choices)
        return data


def validate_values(
    specs: tuple[ParamSpec, ...], values: dict[str, Any], location: str
) -> dict[str, Any]:
    """Check resolved values against a schema; fill defaults; reject unknown names."""
    by_name = {s.name: s for s in specs}
    unknown = sorted(set(values) - set(by_name))
    if unknown:
        raise ProceduralError(
            "UNKNOWN_PARAMETER",
            f"unknown parameter(s): {', '.join(unknown)} (expected: {', '.join(by_name)})",
            location,
        )
    out: dict[str, Any] = {}
    for spec in specs:
        if spec.name not in values:
            if spec.required:
                raise ProceduralError("MISSING_PARAMETER", f"{spec.name!r} is required", location)
            out[spec.name] = spec.default
            continue
        if spec.type in ("any", "operand"):
            out[spec.name] = values[spec.name]
            continue
        try:
            out[spec.name] = check_type(
                spec.type,
                values[spec.name],
                minimum=spec.min,
                maximum=spec.max,
                choices=spec.choices,
            )
        except ProceduralError as exc:
            raise ProceduralError(exc.code, exc.message, f"{location}.{spec.name}") from None
    return out
