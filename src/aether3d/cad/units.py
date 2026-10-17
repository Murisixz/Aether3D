"""Quantities, dimensions and units for precision modeling.

A :class:`Quantity` is a float64 value in SI base units (metres, radians, kilograms) with
a :class:`Dimension` -- the exponents of length, plane angle and mass. Plane angles are
kept as their own dimension (SI treats them as dimensionless) so that a length can never
be passed where an angle is meant, and vice versa.

Conversions are exact: every unit's factor is a rational number (1 in = 25.4 mm and
1 ft = 0.3048 m exactly, by the 1959 international yard and pound agreement; 1 lb =
0.45359237 kg exactly), a parsed value is a rational number too, and the SI value is
their product rounded once to the nearest float64. ``1/16 in`` is therefore the float
nearest to 0.0015875 m, not a product of rounded factors.

Accepted text (``parse_quantity``):

* decimal or scientific numbers, fractions and mixed numbers: ``12.5 mm``, ``1e-3 m``,
  ``3/8 in``, ``2 1/2 in``, ``-0.75 ft``;
* feet and inches: ``3'``, ``3' 4"``, ``3'-4 1/2"``, ``4"``, ``3 ft 4 in``, ``3ft4in``;
* angles: ``45 deg``, ``45°``, ``0.5 rad``, ``30° 15' 10"`` (degrees, minutes, seconds:
  with an expected angle, ``'`` and ``"`` are arc minutes and seconds);
* areas, volumes and masses: ``12 m2``, ``12 m^2``, ``12 m²``, ``3 ft3``, ``2.5 kg``,
  ``2400 kg/m3``.

A bare number takes the default unit given by the caller (the document's display unit).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..errors import CadError

# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dimension:
    """Exponents of length (L), plane angle (A) and mass (M)."""

    length: int = 0
    angle: int = 0
    mass: int = 0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.length + other.length, self.angle + other.angle, self.mass + other.mass
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.length - other.length, self.angle - other.angle, self.mass - other.mass
        )

    def __pow__(self, n: int) -> Dimension:
        return Dimension(self.length * n, self.angle * n, self.mass * n)

    @property
    def name(self) -> str:
        return DIMENSION_NAMES.get(self, self.symbol)

    @property
    def symbol(self) -> str:
        parts = []
        for sym, e in (("L", self.length), ("A", self.angle), ("M", self.mass)):
            if e == 1:
                parts.append(sym)
            elif e:
                parts.append(f"{sym}^{e}")
        return "*".join(parts) or "1"


RATIO = Dimension()
LENGTH = Dimension(length=1)
AREA = Dimension(length=2)
VOLUME = Dimension(length=3)
ANGLE = Dimension(angle=1)
MASS = Dimension(mass=1)
DENSITY = Dimension(length=-3, mass=1)
INERTIA = Dimension(length=2, mass=1)

DIMENSION_NAMES = {
    RATIO: "ratio",
    LENGTH: "length",
    AREA: "area",
    VOLUME: "volume",
    ANGLE: "angle",
    MASS: "mass",
    DENSITY: "density",
    INERTIA: "inertia",
}
DIMENSIONS_BY_NAME = {v: k for k, v in DIMENSION_NAMES.items()}


def article(dim: Dimension) -> str:
    """``"a length"``, ``"an angle"``: a dimension's name for messages."""
    name = dim.name
    return ("an " if name[0] in "aeiou" else "a ") + name


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Unit:
    """A named unit: SI value of one unit (exact rational) and its dimension."""

    symbol: str
    factor: Fraction
    dimension: Dimension
    name: str


_PI = Fraction(math.pi)  # the float pi as a rational: degrees convert like math.radians


def _u(symbol: str, factor: Any, dim: Dimension, name: str) -> Unit:
    return Unit(symbol, Fraction(factor), dim, name)


_LENGTHS = [
    _u("m", 1, LENGTH, "metre"),
    _u("km", 1000, LENGTH, "kilometre"),
    _u("cm", Fraction(1, 100), LENGTH, "centimetre"),
    _u("mm", Fraction(1, 1000), LENGTH, "millimetre"),
    _u("um", Fraction(1, 10**6), LENGTH, "micrometre"),
    _u("in", Fraction(254, 10000), LENGTH, "inch"),
    _u("ft", Fraction(3048, 10000), LENGTH, "foot"),
    _u("yd", Fraction(9144, 10000), LENGTH, "yard"),
]
_ANGLES = [
    _u("rad", 1, ANGLE, "radian"),
    _u("deg", _PI / 180, ANGLE, "degree"),
    _u("arcmin", _PI / (180 * 60), ANGLE, "minute of arc"),
    _u("arcsec", _PI / (180 * 3600), ANGLE, "second of arc"),
    _u("gon", _PI / 200, ANGLE, "gradian"),
]
_MASSES = [
    _u("kg", 1, MASS, "kilogram"),
    _u("g", Fraction(1, 1000), MASS, "gram"),
    _u("t", 1000, MASS, "tonne"),
    _u("lb", Fraction(45359237, 10**8), MASS, "pound"),
]

UNITS: dict[str, Unit] = {}
for _unit in _LENGTHS + _ANGLES + _MASSES:
    UNITS[_unit.symbol] = _unit
for _unit in _LENGTHS:
    for _power, _dim, _word in ((2, AREA, "square"), (3, VOLUME, "cubic")):
        _sym = f"{_unit.symbol}{_power}"
        UNITS[_sym] = Unit(_sym, _unit.factor**_power, _dim, f"{_word} {_unit.name}")
for _mass in _MASSES:
    for _unit in _LENGTHS:
        _sym = f"{_mass.symbol}/{_unit.symbol}3"
        UNITS[_sym] = Unit(
            _sym, _mass.factor / _unit.factor**3, DENSITY, f"{_mass.name} per cubic {_unit.name}"
        )

ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "kilometer": "km",
    "kilometers": "km",
    "µm": "um",
    "μm": "um",
    "inch": "in",
    "inches": "in",
    '"': "in",
    "foot": "ft",
    "feet": "ft",
    "'": "ft",
    "yard": "yd",
    "yards": "yd",
    "°": "deg",
    "degree": "deg",
    "degrees": "deg",
    "radian": "rad",
    "radians": "rad",
    "grad": "gon",
    "gradians": "gon",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "tonne": "t",
    "tonnes": "t",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
}

LENGTH_UNITS = tuple(u.symbol for u in _LENGTHS)
ANGLE_UNITS = tuple(u.symbol for u in _ANGLES)


def unit(symbol: str) -> Unit:
    """The unit with this symbol (``"mm"``, ``"ft2"``, ``"m^3"``, ``"kg/m3"``, ``"°"`` ...)."""
    if not isinstance(symbol, str):
        raise CadError("UNKNOWN_UNIT", f"a unit must be a string, got {type(symbol).__name__}")
    key = _normalize_unit(symbol)
    found = UNITS.get(key)
    if found is None:
        raise CadError(
            "UNKNOWN_UNIT",
            f"unknown unit {symbol!r}",
            hint="lengths: " + ", ".join(LENGTH_UNITS) + "; angles: " + ", ".join(ANGLE_UNITS),
        )
    return found


def _normalize_unit(symbol: str) -> str:
    s = symbol.strip()
    s = ALIASES.get(s, ALIASES.get(s.lower(), s))
    s = s.replace("²", "2").replace("³", "3").replace("^", "")
    for prefix, power in (("sq ", "2"), ("square ", "2"), ("cu ", "3"), ("cubic ", "3")):
        if s.lower().startswith(prefix):
            base = _normalize_unit(s[len(prefix) :])
            return base + power
    if s not in UNITS and s.lower() in UNITS:
        s = s.lower()
    return s


# --------------------------------------------------------------------------- #
# Quantities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Quantity:
    """A value in SI base units with a dimension; immutable.

    Arithmetic checks dimensions: adding a length to an angle raises ``UNIT_MISMATCH``.
    """

    value: float
    dimension: Dimension = LENGTH

    def __post_init__(self) -> None:
        if not isinstance(self.value, float):
            object.__setattr__(self, "value", float(self.value))

    @classmethod
    def of(cls, value: Any, unit_symbol: str) -> Quantity:
        """``value`` (a number) in ``unit_symbol``, converted exactly to SI."""
        u = unit(unit_symbol)
        return cls(_to_si(_rational(value), u), u.dimension)

    def to(self, unit_symbol: str) -> float:
        """The value in another unit of the same dimension."""
        u = unit(unit_symbol)
        self._check(u.dimension, f"cannot express {article(self.dimension)} in {u.symbol}")
        return float(Fraction(self.value) / u.factor)

    def _check(self, dim: Dimension, message: str) -> None:
        if dim != self.dimension:
            raise CadError("UNIT_MISMATCH", message)

    def __add__(self, other: Any) -> Quantity:
        other = _as_quantity(other)
        if other.dimension != self.dimension:
            raise CadError(
                "UNIT_MISMATCH",
                f"cannot add {article(other.dimension)} to {article(self.dimension)}",
            )
        return Quantity(self.value + other.value, self.dimension)

    def __sub__(self, other: Any) -> Quantity:
        other = _as_quantity(other)
        if other.dimension != self.dimension:
            raise CadError(
                "UNIT_MISMATCH",
                f"cannot subtract {article(other.dimension)} from {article(self.dimension)}",
            )
        return Quantity(self.value - other.value, self.dimension)

    def __mul__(self, other: Any) -> Quantity:
        other = _as_quantity(other)
        return Quantity(self.value * other.value, self.dimension * other.dimension)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Quantity:
        other = _as_quantity(other)
        if other.value == 0.0:
            raise CadError("DIVISION_BY_ZERO", "division by zero")
        return Quantity(self.value / other.value, self.dimension / other.dimension)

    def __rtruediv__(self, other: Any) -> Quantity:
        return _as_quantity(other) / self

    def __neg__(self) -> Quantity:
        return Quantity(-self.value, self.dimension)

    def __abs__(self) -> Quantity:
        return Quantity(abs(self.value), self.dimension)

    def __pow__(self, n: Any) -> Quantity:
        if isinstance(n, Quantity):
            if n.dimension != RATIO:
                raise CadError("UNIT_MISMATCH", "an exponent must be a plain number")
            n = n.value
        if self.dimension != RATIO:
            if float(n) != int(n) or abs(int(n)) > 4:
                raise CadError(
                    "UNIT_MISMATCH",
                    f"{article(self.dimension)} can only be raised to an integer power -4..4",
                )
            n = int(n)
            return Quantity(self.value**n, self.dimension**n)
        return Quantity(self.value ** float(n), RATIO)

    def _compare(self, other: Any) -> tuple[float, float]:
        other = _as_quantity(other)
        if other.dimension != self.dimension:
            raise CadError(
                "UNIT_MISMATCH",
                f"cannot compare {article(self.dimension)} with {article(other.dimension)}",
            )
        return self.value, other.value

    def __lt__(self, other: Any) -> bool:
        a, b = self._compare(other)
        return a < b

    def __le__(self, other: Any) -> bool:
        a, b = self._compare(other)
        return a <= b

    def __gt__(self, other: Any) -> bool:
        a, b = self._compare(other)
        return a > b

    def __ge__(self, other: Any) -> bool:
        a, b = self._compare(other)
        return a >= b

    def is_finite(self) -> bool:
        return math.isfinite(self.value)

    def format(self, unit_symbol: str | None = None, **options: Any) -> str:
        return format_quantity(self, unit_symbol, **options)

    def __str__(self) -> str:
        return format_quantity(self)


def _as_quantity(value: Any) -> Quantity:
    if isinstance(value, Quantity):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CadError("INVALID_QUANTITY", f"expected a number or quantity, got {value!r}")
    return Quantity(float(value), RATIO)


def _rational(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CadError("INVALID_QUANTITY", f"expected a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise CadError("INVALID_QUANTITY", f"{value} is not a finite number")
    return Fraction(value)


def _to_si(value: Fraction, u: Unit) -> float:
    exact = value * u.factor
    try:
        out = float(exact)
    except OverflowError:
        raise CadError("INVALID_QUANTITY", "the value is too large") from None
    return out


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

MAX_TEXT = 200
_NUMBER = r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?|\.[0-9]+(?:[eE][+-]?[0-9]+)?"
# A mixed number "2 1/2", a fraction "1/2" or a decimal number.
_VALUE = rf"(?:[0-9]+\s+[0-9]+/[0-9]+|[0-9]+/[0-9]+|{_NUMBER})"
_RE_VALUE_UNIT = re.compile(rf"\s*({_VALUE})\s*([A-Za-zµμ°'\"²³^0-9/ ]*?)\s*$")
_RE_FEET_INCHES = re.compile(
    rf"""^\s*(?:(?P<ft>{_VALUE})\s*(?:'|ft|feet|foot)\s*)?
          (?:-?\s*(?P<in>{_VALUE})\s*(?:"|in|inch|inches)\s*)?$""",
    re.VERBOSE,
)
_RE_DMS = re.compile(
    rf"""^\s*(?:(?P<d>{_VALUE})\s*(?:°|deg)\s*)?
          (?:(?P<m>{_VALUE})\s*(?:'|arcmin)\s*)?
          (?:(?P<s>{_VALUE})\s*(?:"|arcsec)\s*)?$""",
    re.VERBOSE,
)


def _parse_value(text: str) -> Fraction:
    text = text.strip()
    parts = text.split()
    try:
        if len(parts) == 2 and "/" in parts[1]:
            whole = Fraction(parts[0])
            frac = Fraction(parts[1])
            return whole + frac
        if len(parts) == 1:
            return Fraction(parts[0])
    except (ValueError, ZeroDivisionError):
        pass
    raise CadError("INVALID_QUANTITY", f"{text!r} is not a number")


def parse_quantity(
    text: Any, *, expected: Dimension | None = None, default_unit: str | None = None
) -> Quantity:
    """Parse ``text`` (a string, or a number in ``default_unit``) into a :class:`Quantity`.

    ``expected`` (e.g. :data:`LENGTH`) checks the dimension and decides what ``'`` and ``"``
    mean (feet and inches for lengths, arc minutes and seconds for angles).
    """
    if isinstance(text, Quantity):
        q = text
    elif isinstance(text, (int, float)) and not isinstance(text, bool):
        if default_unit is None:
            if expected not in (None, RATIO):
                raise CadError(
                    "UNIT_REQUIRED",
                    f"the number {text!r} needs a unit ({article(expected)} is expected)",
                    hint="write it as a string with a unit, e.g. '12 mm'",
                )
            q = Quantity(_to_si(_rational(text), Unit("", Fraction(1), RATIO, "")), RATIO)
        else:
            q = Quantity.of(text, default_unit)
    elif isinstance(text, str):
        q = _parse_text(text, expected, default_unit)
    else:
        raise CadError("INVALID_QUANTITY", f"expected a quantity, got {type(text).__name__}")
    if not q.is_finite():
        raise CadError("INVALID_QUANTITY", f"{text!r} is not finite")
    if expected is not None and q.dimension != expected:
        raise CadError(
            "UNIT_MISMATCH",
            f"{text!r} is {article(q.dimension)}; {article(expected)} is expected",
        )
    return q


def _parse_text(text: str, expected: Dimension | None, default_unit: str | None) -> Quantity:
    if len(text) > MAX_TEXT:
        raise CadError("INVALID_QUANTITY", f"quantity text longer than {MAX_TEXT} characters")
    s = text.strip()
    sign = 1
    if s.startswith(("-", "+")):
        sign = -1 if s[0] == "-" else 1
        s = s[1:].strip()
    if not s:
        raise CadError("INVALID_QUANTITY", f"{text!r} is not a quantity")
    q = None
    if expected == ANGLE or (expected is None and ("°" in s or "deg" in s)):
        m = _RE_DMS.match(s)
        if m and any(m.group(g) for g in ("d", "m", "s")):
            total = Fraction(0)
            for g, div in (("d", 1), ("m", 60), ("s", 3600)):
                if m.group(g):
                    total += _parse_value(m.group(g)) / div
            q = Quantity(_to_si(total, UNITS["deg"]), ANGLE)
    if q is None and expected != ANGLE:
        m = _RE_FEET_INCHES.match(s)
        if m and (m.group("ft") or m.group("in")):
            total = Fraction(0)
            if m.group("ft"):
                total += _parse_value(m.group("ft")) * 12
            if m.group("in"):
                total += _parse_value(m.group("in"))
            q = Quantity(_to_si(total, UNITS["in"]), LENGTH)
    if q is None:
        m = _RE_VALUE_UNIT.match(s)
        if not m:
            raise CadError("INVALID_QUANTITY", f"{text!r} is not a quantity")
        value = _parse_value(m.group(1))
        symbol = m.group(2).strip()
        if not symbol:
            if default_unit is None:
                if expected not in (None, RATIO):
                    raise CadError(
                        "UNIT_REQUIRED",
                        f"{text!r} needs a unit ({article(expected)} is expected)",
                        hint="for example '12 mm', '3 ft 4 in' or '45 deg'",
                    )
                q = Quantity(_to_si(value, Unit("", Fraction(1), RATIO, "")), RATIO)
            else:
                u = unit(default_unit)
                q = Quantity(_to_si(value, u), u.dimension)
        else:
            u = unit(symbol)
            q = Quantity(_to_si(value, u), u.dimension)
    return Quantity(sign * q.value, q.dimension)


def parse_length(text: Any, default_unit: str | None = None) -> float:
    """A length in metres from text or a number in ``default_unit``."""
    return parse_quantity(text, expected=LENGTH, default_unit=default_unit).value


def parse_angle(text: Any, default_unit: str | None = "deg") -> float:
    """An angle in radians from text or a number in ``default_unit`` (degrees)."""
    return parse_quantity(text, expected=ANGLE, default_unit=default_unit).value


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

DEFAULT_UNITS = {
    LENGTH: "mm",
    AREA: "m2",
    VOLUME: "m3",
    ANGLE: "deg",
    MASS: "kg",
    DENSITY: "kg/m3",
    RATIO: "",
}


def format_quantity(
    q: Quantity,
    unit_symbol: str | None = None,
    *,
    precision: int = 3,
    style: str = "decimal",
    denominator: int = 16,
) -> str:
    """Display text for ``q``.

    * ``style="decimal"``: ``precision`` decimals, trailing zeros removed (``12.5 mm``);
    * ``style="architectural"`` (lengths): feet, inches and a fraction rounded to
      ``1/denominator`` inch (``3'-4 1/2"``);
    * ``style="fractional"`` (lengths): inches and a fraction (``40 1/2"``).

    Rounding is to nearest, ties to even, of the exact binary value.
    """
    if not isinstance(q, Quantity):
        q = _as_quantity(q)
    if style in ("architectural", "fractional"):
        if q.dimension != LENGTH:
            raise CadError("UNIT_MISMATCH", f"{style} style formats lengths only")
        return _format_feet_inches(q.value, style, denominator)
    if style != "decimal":
        raise CadError(
            "INVALID_FORMAT", f"unknown style {style!r}; use decimal, architectural, fractional"
        )
    if not isinstance(precision, int) or not 0 <= precision <= 15:
        raise CadError("INVALID_FORMAT", "precision must be an integer 0..15")
    symbol = DEFAULT_UNITS.get(q.dimension, "") if unit_symbol is None else unit_symbol
    value = q.to(symbol) if symbol else q.value
    text = f"{value:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    if not symbol:
        return text
    display = {"deg": "°", "m2": "m²", "m3": "m³", "mm2": "mm²", "ft2": "ft²"}.get(symbol, symbol)
    return f"{text}°" if display == "°" else f"{text} {display}"


def _format_feet_inches(metres: float, style: str, denominator: int) -> str:
    if not isinstance(denominator, int) or denominator not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise CadError("INVALID_FORMAT", "denominator must be a power of two 1..128")
    inches = Fraction(metres) / Fraction(254, 10000)
    sign = "-" if inches < 0 else ""
    units = abs(inches) * denominator
    n = round(units)  # Fraction rounding: nearest, ties to even
    whole_in, num = divmod(n, denominator)
    frac = Fraction(num, denominator)
    feet, inch = divmod(whole_in, 12) if style == "architectural" else (0, whole_in)
    if frac:
        inch_text = (
            f"{inch} {frac.numerator}/{frac.denominator}"
            if inch
            else (f"{frac.numerator}/{frac.denominator}")
        )
    else:
        inch_text = f"{inch}"
    if style == "architectural":
        return f"{sign}{feet}'-{inch_text}\""
    return f'{sign}{inch_text}"'


def quantity_to_json(q: Quantity) -> dict[str, Any]:
    return {"value": q.value, "dimension": q.dimension.name}


def describe_units() -> dict[str, Any]:
    """Every unit by dimension (for discovery by agents)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for u in UNITS.values():
        out.setdefault(u.dimension.name, []).append(
            {"symbol": u.symbol, "name": u.name, "si": float(u.factor)}
        )
    return {
        "dimensions": out,
        "aliases": dict(sorted(ALIASES.items())),
        "exact_definitions": {
            "in": "0.0254 m",
            "ft": "0.3048 m",
            "yd": "0.9144 m",
            "lb": "0.45359237 kg",
            "deg": "pi/180 rad (pi as float64)",
        },
    }
