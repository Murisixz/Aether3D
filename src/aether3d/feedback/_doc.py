"""Strict checking of the feedback package's JSON documents.

Expectations, view specifications, vision evaluations, correction actions, policies and
budgets all arrive as untrusted JSON. :class:`Doc` reads them with typed accessors that
record every problem (code, message, JSON pointer, hint) instead of stopping at the
first, as scene plans do; :meth:`Issues.raise_if_any` then raises one
:class:`~aether3d.errors.FeedbackError` that lists them all. Nothing in a document is ever
executed: values are numbers, strings, enumerations and object selectors.
"""

from __future__ import annotations

import difflib
import math
from typing import Any

from ..errors import FeedbackError

BAD = object()
"""Returned by an accessor whose value was invalid (the problem is recorded)."""

MAX_TEXT = 2000
"""Longest free text (summaries, rationales) a document may carry."""


def escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def join(pointer: str, key: str | int) -> str:
    return f"{pointer}/{escape(key) if isinstance(key, str) else key}"


def kind(value: Any) -> str:
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


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Issues:
    """Problems found in one document (the first ``limit`` are kept, all are counted)."""

    def __init__(self, limit: int = 25) -> None:
        self.items: list[dict[str, Any]] = []
        self.limit = limit
        self.total = 0

    def add(self, code: str, message: str, pointer: str, *, hint: str | None = None) -> None:
        self.total += 1
        if len(self.items) < self.limit:
            self.items.append(
                {"code": code, "message": message, "pointer": pointer or "", "hint": hint}
            )

    def __bool__(self) -> bool:
        return self.total > 0

    def raise_if_any(self, what: str) -> None:
        if not self.total:
            return
        first = self.items[0]
        extra = {} if self.total <= len(self.items) else {"omitted": self.total - len(self.items)}
        raise FeedbackError(
            first["code"],
            f"{what}: {first['message']}",
            first["pointer"] or None,
            hint=first["hint"],
            details=extra,
            issues=self.items,
        )


class Doc:
    """Typed accessors over an untrusted JSON document (see the module docstring)."""

    def __init__(self, issues: Issues | None = None) -> None:
        self.issues = issues if issues is not None else Issues()

    # -- structure ---------------------------------------------------------

    def object(self, value: Any, ptr: str, what: str = "an object") -> dict | None:
        if not isinstance(value, dict):
            self.issues.add("INVALID_TYPE", f"expected {what}, got {kind(value)}", ptr)
            return None
        return value

    def keys(self, obj: dict, ptr: str, allowed: tuple[str, ...], required: tuple = ()) -> bool:
        ok = True
        for key in obj:
            if key not in allowed:
                close = (
                    difflib.get_close_matches(key, list(allowed), n=1, cutoff=0.75)
                    if isinstance(key, str)
                    else []
                )
                self.issues.add(
                    "UNKNOWN_KEY",
                    f"unknown key {key!r}",
                    join(ptr, str(key)),
                    hint=f"did you mean {close[0]!r}?"
                    if close
                    else f"allowed: {', '.join(allowed)}",
                )
                ok = False
        for key in required:
            if key not in obj:
                self.issues.add("MISSING_KEY", f"{key!r} is required", join(ptr, key))
                ok = False
        return ok

    def array(
        self, value: Any, ptr: str, *, min_items: int = 0, max_items: int = 1000
    ) -> list | None:
        if not isinstance(value, list):
            self.issues.add("INVALID_TYPE", f"expected an array, got {kind(value)}", ptr)
            return None
        if len(value) < min_items:
            self.issues.add("INVALID_VALUE", f"needs at least {min_items} item(s)", ptr)
            return None
        if len(value) > max_items:
            self.issues.add("LIMIT_EXCEEDED", f"at most {max_items} items", ptr)
            return None
        return value

    # -- scalars -----------------------------------------------------------

    def string(self, value: Any, ptr: str, *, max_length: int = 256, empty: bool = False) -> Any:
        if not isinstance(value, str) or (not empty and not value.strip()):
            self.issues.add("INVALID_VALUE", f"expected a non-empty string, got {kind(value)}", ptr)
            return BAD
        if len(value) > max_length:
            self.issues.add("LIMIT_EXCEEDED", f"string longer than {max_length} characters", ptr)
            return BAD
        if any(ord(c) < 32 and c not in "\t\n" for c in value):
            self.issues.add("INVALID_VALUE", "control characters are not allowed", ptr)
            return BAD
        return value

    def text(self, value: Any, ptr: str) -> Any:
        """Free text (a summary, a rationale): bounded, never interpreted."""
        return self.string(value, ptr, max_length=MAX_TEXT, empty=True)

    def boolean(self, value: Any, ptr: str) -> Any:
        if not isinstance(value, bool):
            self.issues.add("INVALID_VALUE", f"expected true or false, got {kind(value)}", ptr)
            return BAD
        return value

    def number(
        self,
        value: Any,
        ptr: str,
        *,
        lo: float | None = None,
        hi: float | None = None,
        lo_open: bool = False,
    ) -> Any:
        if not is_number(value) or not math.isfinite(value):
            self.issues.add("INVALID_VALUE", f"expected a finite number, got {kind(value)}", ptr)
            return BAD
        below = lo is not None and (value <= lo if lo_open else value < lo)
        if below or (hi is not None and value > hi):
            low = "-inf" if lo is None else f"{lo:g}"
            self.issues.add(
                "OUT_OF_RANGE",
                f"{value:g} is outside {'(' if lo_open else '['}{low}, "
                f"{'inf' if hi is None else f'{hi:g}'}]",
                ptr,
            )
            return BAD
        return float(value)

    def integer(self, value: Any, ptr: str, *, lo: int | None = None, hi: int | None = None) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                self.issues.add("INVALID_VALUE", f"expected an integer, got {kind(value)}", ptr)
                return BAD
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            self.issues.add(
                "OUT_OF_RANGE",
                f"{value} is outside [{'-inf' if lo is None else lo}, "
                f"{'inf' if hi is None else hi}]",
                ptr,
            )
            return BAD
        return value

    def enum(self, value: Any, ptr: str, choices: tuple[str, ...]) -> Any:
        if not isinstance(value, str) or value not in choices:
            close = (
                difflib.get_close_matches(value, list(choices), n=1, cutoff=0.7)
                if isinstance(value, str)
                else []
            )
            self.issues.add(
                "INVALID_VALUE",
                f"expected one of {', '.join(choices)}, got {kind(value)}",
                ptr,
                hint=f"did you mean {close[0]!r}?" if close else None,
            )
            return BAD
        return value

    def vector(self, value: Any, ptr: str, size: int = 3, *, nonzero: bool = False) -> Any:
        if not isinstance(value, (list, tuple)) or len(value) != size:
            names = "[x, y, z]" if size == 3 else f"{size} numbers"
            self.issues.add("INVALID_VALUE", f"expected {names}, got {kind(value)}", ptr)
            return BAD
        items = [self.number(v, join(ptr, k)) for k, v in enumerate(value)]
        if any(v is BAD for v in items):
            return BAD
        if nonzero and not any(items):
            self.issues.add("INVALID_VALUE", "the vector must not be zero", ptr)
            return BAD
        return tuple(items)

    def range(
        self, value: Any, ptr: str, *, lo: float | None = None, hi: float | None = None
    ) -> Any:
        """``{"min": a, "max": b}`` (either may be left out) -> ``(min | None, max | None)``."""
        obj = self.object(value, ptr, 'a range {"min": a, "max": b}')
        if obj is None:
            return BAD
        if not self.keys(obj, ptr, ("min", "max")):
            return BAD
        if not obj:
            self.issues.add("INVALID_VALUE", "a range needs min, max or both", ptr)
            return BAD
        a = None if "min" not in obj else self.number(obj["min"], join(ptr, "min"), lo=lo, hi=hi)
        b = None if "max" not in obj else self.number(obj["max"], join(ptr, "max"), lo=lo, hi=hi)
        if a is BAD or b is BAD:
            return BAD
        if a is not None and b is not None and a > b:
            self.issues.add("INVALID_VALUE", "min is greater than max", ptr)
            return BAD
        return (a, b)

    def rect(self, value: Any, ptr: str, *, normalized: bool) -> Any:
        """``[x0, y0, x1, y1]`` with ``x0 < x1``, ``y0 < y1`` (in 0..1 when normalized)."""
        v = self.vector(value, ptr, 4)
        if v is BAD:
            return BAD
        x0, y0, x1, y1 = v
        if not (x0 < x1 and y0 < y1):
            self.issues.add("INVALID_VALUE", "a rectangle needs x0 < x1 and y0 < y1", ptr)
            return BAD
        if normalized and not (x0 >= 0.0 and y0 >= 0.0 and x1 <= 1.0 and y1 <= 1.0):
            self.issues.add(
                "OUT_OF_RANGE",
                "a normalized rectangle lies in [0, 1] (fractions of the image width and height)",
                ptr,
            )
            return BAD
        return v

    def color(self, value: Any, ptr: str) -> Any:
        from ..rendering.color import parse_color

        try:
            parse_color(value, ptr)
        except Exception as exc:  # RenderError
            self.issues.add("INVALID_VALUE", getattr(exc, "message", str(exc)), ptr)
            return BAD
        return value

    def selector(self, value: Any, ptr: str) -> Any:
        """An object selection, as scene plans and the MCP tools take it: an id or name, a
        list of them, or ``{"tags": [...], "name_glob": ..., ...}``. Checked for shape here;
        resolved against a scene later (:func:`aether3d.plans.select_objects`)."""
        from ..plans import PlanLimits
        from ..plans.model import Issues as PlanIssues
        from ..plans.model import Reader, SymbolRef, parse_selector

        found = PlanIssues()
        sel = parse_selector(Reader(found, PlanLimits()), value, ptr)
        for item in found.items:
            self.issues.add(item["code"], item["message"], item["pointer"], hint=item.get("hint"))
        if sel is None or found.total:
            return BAD
        for k, ref in enumerate(sel.refs):
            if isinstance(ref, SymbolRef):
                self.issues.add(
                    "INVALID_REFERENCE",
                    f"{ref.text!r}: plan symbols exist only inside a plan",
                    join(ptr, k) if isinstance(value, list) else ptr,
                    hint="name objects by id or name",
                )
                return BAD
        return value
