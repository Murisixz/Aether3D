"""Unit-aware expressions for parameters and dimensions (data, never code).

An expression is text such as ``"width / 2 - 10 mm"``, ``"2 * wall + 3' 4\\""`` or
``"atan2(rise, run)"``. It is parsed with Python's ``ast`` module in ``eval`` mode and
walked by a fixed interpreter that accepts only:

* number literals, and quantity literals written as a number directly followed by a unit
  (``10 mm``, ``2.5 m``, ``45 deg``) -- inside an expression a fraction needs parentheses:
  ``(1/2) in``;
* names of parameters (resolved by the caller) and the constant ``pi``;
* ``+ - * / ** %``, unary ``-``/``+``, comparisons, ``and``/``or``/``not`` and
  ``a if condition else b``;
* calls of the listed functions: ``abs min max sqrt hypot sin cos tan asin acos atan
  atan2 floor ceil round radians degrees clamp``.

Nothing else is evaluated: no attribute access, no subscripts, no other calls, no
comprehensions, no lambdas. ``eval``/``exec`` are never used. Sizes are bounded (text
length, AST nodes, nesting) and every intermediate value must be finite.

Units are checked: ``width + 45 deg`` raises ``UNIT_MISMATCH``; trigonometric functions
take angles; ``floor``/``ceil``/``round`` take plain numbers (``round(width / 1 mm)``).
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from ..errors import CadError
from .units import ANGLE, RATIO, UNITS, Quantity, _normalize_unit, article, parse_quantity

MAX_LENGTH = 1000
MAX_NODES = 300
MAX_DEPTH = 40
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# A number directly followed by a unit symbol (longest symbols first so "mm" wins over "m").
_UNIT_SYMBOLS = sorted(
    {s for s in UNITS if "/" not in s} | {"°", "ft2", "in2", "m2", "mm2", "m3"},
    key=len,
    reverse=True,
)
_NUM = r"(?<![A-Za-z_0-9.])((?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
_QUANTITY_LITERAL = re.compile(
    _NUM + r"\s*(" + "|".join(re.escape(s) for s in _UNIT_SYMBOLS) + r")(?![A-Za-z_0-9])"
)
_GROUP_UNIT = re.compile(
    r"\)\s*(" + "|".join(re.escape(s) for s in _UNIT_SYMBOLS) + r")(?![A-Za-z_0-9])"
)
_FEET_INCHES = re.compile(
    r"(?<![A-Za-z_0-9.])([0-9]+(?:\.[0-9]+)?)\s*'\s*-?\s*(?:([0-9]+(?:\.[0-9]+)?)\s*\")?"
    r"|(?<![A-Za-z_0-9.'])([0-9]+(?:\.[0-9]+)?)\s*\""
)


def _q(x: Any) -> Quantity:
    if isinstance(x, Quantity):
        return x
    return Quantity(float(x), RATIO)


def _ratio(x: Quantity, fn: str) -> float:
    if x.dimension != RATIO:
        raise CadError("UNIT_MISMATCH", f"{fn}() takes a plain number, got {article(x.dimension)}")
    return x.value


def _angle(x: Quantity, fn: str) -> float:
    if x.dimension != ANGLE:
        raise CadError("UNIT_MISMATCH", f"{fn}() takes an angle, got {article(x.dimension)}")
    return x.value


def _same(xs: tuple[Quantity, ...], fn: str) -> None:
    for x in xs[1:]:
        if x.dimension != xs[0].dimension:
            raise CadError("UNIT_MISMATCH", f"{fn}() arguments must have the same unit kind")


def _min(*xs: Quantity) -> Quantity:
    _same(xs, "min")
    return min(xs, key=lambda q: q.value)


def _max(*xs: Quantity) -> Quantity:
    _same(xs, "max")
    return max(xs, key=lambda q: q.value)


def _sqrt(x: Quantity) -> Quantity:
    d = x.dimension
    if d.length % 2 or d.angle % 2 or d.mass % 2:
        raise CadError("UNIT_MISMATCH", f"sqrt() of {article(d)} has no unit")
    if x.value < 0:
        raise CadError("MATH_DOMAIN", "sqrt() of a negative value")
    from .units import Dimension

    return Quantity(math.sqrt(x.value), Dimension(d.length // 2, d.angle // 2, d.mass // 2))


def _hypot(*xs: Quantity) -> Quantity:
    _same(xs, "hypot")
    return Quantity(math.hypot(*(x.value for x in xs)), xs[0].dimension)


def _inverse(fn: Callable[[float], float], name: str) -> Callable[[Quantity], Quantity]:
    def f(x: Quantity) -> Quantity:
        v = _ratio(x, name)
        try:
            return Quantity(fn(v), ANGLE)
        except ValueError:
            raise CadError("MATH_DOMAIN", f"{name}({v:g}) is undefined") from None

    return f


def _atan2(y: Quantity, x: Quantity) -> Quantity:
    _same((y, x), "atan2")
    return Quantity(math.atan2(y.value, x.value), ANGLE)


def _clamp(x: Quantity, lo: Quantity, hi: Quantity) -> Quantity:
    _same((x, lo, hi), "clamp")
    return Quantity(min(max(x.value, lo.value), hi.value), x.dimension)


FUNCTIONS: dict[str, tuple[Callable[..., Quantity], int, int]] = {
    "abs": (lambda x: abs(x), 1, 1),
    "min": (_min, 1, 64),
    "max": (_max, 1, 64),
    "sqrt": (_sqrt, 1, 1),
    "hypot": (_hypot, 1, 8),
    "sin": (lambda x: Quantity(math.sin(_angle(x, "sin")), RATIO), 1, 1),
    "cos": (lambda x: Quantity(math.cos(_angle(x, "cos")), RATIO), 1, 1),
    "tan": (lambda x: Quantity(math.tan(_angle(x, "tan")), RATIO), 1, 1),
    "asin": (_inverse(math.asin, "asin"), 1, 1),
    "acos": (_inverse(math.acos, "acos"), 1, 1),
    "atan": (_inverse(math.atan, "atan"), 1, 1),
    "atan2": (_atan2, 2, 2),
    "floor": (lambda x: Quantity(float(math.floor(_ratio(x, "floor"))), RATIO), 1, 1),
    "ceil": (lambda x: Quantity(float(math.ceil(_ratio(x, "ceil"))), RATIO), 1, 1),
    "round": (lambda x: Quantity(float(round(_ratio(x, "round"))), RATIO), 1, 1),
    "radians": (lambda x: Quantity(math.radians(_ratio(x, "radians")), ANGLE), 1, 1),
    "degrees": (lambda x: Quantity(math.degrees(_angle(x, "degrees")), RATIO), 1, 1),
    "clamp": (_clamp, 3, 3),
}
CONSTANTS = {"pi": Quantity(math.pi, RATIO)}

_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


class Expression:
    """A parsed, checked expression; ``names`` are the parameters it reads."""

    __slots__ = ("source", "_tree", "names", "_literals")

    def __init__(self, source: str) -> None:
        if not isinstance(source, str):
            raise CadError("INVALID_EXPRESSION", "an expression must be a string")
        if len(source) > MAX_LENGTH:
            raise CadError("INVALID_EXPRESSION", f"expression longer than {MAX_LENGTH} characters")
        self.source = source
        text, self._literals = _rewrite_literals(source)
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as exc:
            raise CadError("INVALID_EXPRESSION", f"cannot parse {source!r}: {exc.msg}") from None
        self.names = frozenset(_check(tree, set(self._literals)))
        self._tree = tree

    def evaluate(self, resolve: Mapping[str, Any] | Callable[[str], Any]) -> Quantity:
        """The value, reading parameters through ``resolve`` (a mapping or a function)."""
        get = resolve.__getitem__ if isinstance(resolve, Mapping) else resolve
        result = _Eval(get, self._literals).run(self._tree.body)
        if isinstance(result, bool):
            return Quantity(1.0 if result else 0.0, RATIO)
        return result

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expression) and other.source == self.source

    def __hash__(self) -> int:
        return hash(self.source)


def _rewrite_literals(source: str) -> tuple[str, dict[str, Quantity]]:
    """Replace quantity literals by placeholder names bound to their values."""
    literals: dict[str, Quantity] = {}

    def bind(q: Quantity) -> str:
        name = f"__q{len(literals)}"
        literals[name] = q
        return f" {name} "

    def feet_inches(m: re.Match) -> str:
        return bind(parse_quantity(m.group(0).strip(), default_unit=None))

    text = _FEET_INCHES.sub(feet_inches, source)

    def quantity(m: re.Match) -> str:
        symbol = _normalize_unit(m.group(2))
        return bind(parse_quantity(f"{m.group(1)} {symbol}"))

    text = _QUANTITY_LITERAL.sub(quantity, text)

    def group(m: re.Match) -> str:
        symbol = _normalize_unit(m.group(1))
        return ") *" + bind(parse_quantity(f"1 {symbol}"))

    text = _GROUP_UNIT.sub(group, text)
    return text, literals


def _check(tree: ast.AST, literals: set[str]) -> set[str]:
    names: set[str] = set()
    count = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal count
        count += 1
        if count > MAX_NODES:
            raise CadError("INVALID_EXPRESSION", f"expression has more than {MAX_NODES} parts")
        if depth > MAX_DEPTH:
            raise CadError("INVALID_EXPRESSION", "expression is nested too deeply")
        if isinstance(node, ast.Expression):
            visit(node.body, depth + 1)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise CadError("INVALID_EXPRESSION", f"unsupported literal {node.value!r}")
            return
        if isinstance(node, ast.Name):
            if node.id in literals or node.id in CONSTANTS:
                return
            if node.id.startswith("_") or not _NAME.match(node.id):
                raise CadError("INVALID_EXPRESSION", f"invalid name {node.id!r}")
            names.add(node.id)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINOPS):
                raise CadError("INVALID_EXPRESSION", "unsupported operator")
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
                raise CadError("INVALID_EXPRESSION", "unsupported operator")
            visit(node.operand, depth + 1)
            return
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                visit(v, depth + 1)
            return
        if isinstance(node, ast.Compare):
            if not all(isinstance(op, _CMPOPS) for op in node.ops):
                raise CadError("INVALID_EXPRESSION", "unsupported comparison")
            visit(node.left, depth + 1)
            for c in node.comparators:
                visit(c, depth + 1)
            return
        if isinstance(node, ast.IfExp):
            for part in (node.test, node.body, node.orelse):
                visit(part, depth + 1)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise CadError(
                    "INVALID_EXPRESSION",
                    "only these functions may be called: " + ", ".join(sorted(FUNCTIONS)),
                )
            if node.keywords:
                raise CadError("INVALID_EXPRESSION", "keyword arguments are not allowed")
            _, lo, hi = FUNCTIONS[node.func.id]
            if not lo <= len(node.args) <= hi:
                raise CadError("INVALID_EXPRESSION", f"{node.func.id}() takes {lo}..{hi} arguments")
            for a in node.args:
                if isinstance(a, ast.Starred):
                    raise CadError("INVALID_EXPRESSION", "starred arguments are not allowed")
                visit(a, depth + 1)
            return
        raise CadError("INVALID_EXPRESSION", f"unsupported syntax: {type(node).__name__}")

    visit(tree, 0)
    return names


class _Eval:
    def __init__(self, get: Callable[[str], Any], literals: dict[str, Quantity]) -> None:
        self.get = get
        self.literals = literals

    def run(self, node: ast.AST) -> Any:
        value = self._run(node)
        if isinstance(value, Quantity) and not value.is_finite():
            raise CadError("MATH_DOMAIN", "the expression's value is not finite")
        return value

    def _num(self, node: ast.AST) -> Quantity:
        v = self.run(node)
        if isinstance(v, bool):
            raise CadError("INVALID_EXPRESSION", "a comparison is not a number")
        return v

    def _run(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return Quantity(float(node.value), RATIO)
        if isinstance(node, ast.Name):
            if node.id in self.literals:
                return self.literals[node.id]
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            try:
                value = self.get(node.id)
            except KeyError:
                raise CadError("UNKNOWN_PARAMETER", f"unknown parameter {node.id!r}") from None
            if isinstance(value, Quantity):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return Quantity(float(value), RATIO)
            raise CadError("INVALID_EXPRESSION", f"parameter {node.id!r} is not a number")
        if isinstance(node, ast.BinOp):
            a, b = self._num(node.left), self._num(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, ast.Mod):
                if a.dimension != b.dimension:
                    raise CadError("UNIT_MISMATCH", "% needs operands of the same unit kind")
                if b.value == 0:
                    raise CadError("DIVISION_BY_ZERO", "modulo by zero")
                return Quantity(math.fmod(a.value, b.value), a.dimension)
            if abs(b.value) > 64 and a.dimension == RATIO:
                raise CadError("MATH_DOMAIN", "exponent too large")
            try:
                return a**b
            except OverflowError:
                raise CadError("MATH_DOMAIN", "power overflows") from None
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self._truth(node.operand)
            v = self._num(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._truth(v) for v in node.values)
            return any(self._truth(v) for v in node.values)
        if isinstance(node, ast.Compare):
            left = self._num(node.left)
            for op, comp in zip(node.ops, node.comparators, strict=True):
                right = self._num(comp)
                a, b = left._compare(right)
                ok = {
                    ast.Lt: a < b,
                    ast.LtE: a <= b,
                    ast.Gt: a > b,
                    ast.GtE: a >= b,
                    ast.Eq: a == b,
                    ast.NotEq: a != b,
                }[type(op)]
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.run(node.body) if self._truth(node.test) else self.run(node.orelse)
        if isinstance(node, ast.Call):
            fn = FUNCTIONS[node.func.id][0]
            try:
                return fn(*(self._num(a) for a in node.args))
            except OverflowError:
                raise CadError("MATH_DOMAIN", f"{node.func.id}() overflows") from None
        raise CadError("INVALID_EXPRESSION", "unsupported syntax")  # pragma: no cover

    def _truth(self, node: ast.AST) -> bool:
        v = self.run(node)
        if isinstance(v, bool):
            return v
        return v.value != 0.0


def is_expression(value: Any) -> bool:
    """Text that is an expression rather than a plain quantity (``"=..."`` or has names)."""
    return isinstance(value, str) and value.strip().startswith("=")


def evaluate_value(
    value: Any,
    resolve: Mapping[str, Any] | Callable[[str], Any],
    *,
    expected: Any = None,
    default_unit: str | None = None,
) -> Quantity:
    """A literal (number in ``default_unit``, quantity text) or an ``"=expression"``."""
    if is_expression(value):
        q = Expression(value.strip()[1:]).evaluate(resolve)
        if expected is not None and q.dimension != expected:
            if q.dimension == RATIO and default_unit is not None:
                return parse_quantity(q.value, expected=expected, default_unit=default_unit)
            raise CadError(
                "UNIT_MISMATCH",
                f"{value!r} is {article(q.dimension)}; {article(expected)} is expected",
            )
        return q
    return parse_quantity(value, expected=expected, default_unit=default_unit)
