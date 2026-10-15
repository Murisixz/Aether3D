"""Switches between an implementation and the one it replaces (performance work).

Every performance change that replaces a working implementation keeps the old one
available behind a switch until the new one has proven itself: tests run both and
compare them (identical results), benchmarks compare their speed, and anyone can
switch back without a new release. A switch is an environment variable, read every
time it is consulted (no restart needed in tests), registered in :data:`SWITCHES`
with its default and documented in ``docs/PERFORMANCE.md``.

Values: ``1``/``on``/``true``/``yes`` and ``0``/``off``/``false``/``no`` (any case);
unset or empty means the default. Anything else raises ``ValueError`` naming the
variable, so a typo never silently selects an implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUE = frozenset({"1", "on", "true", "yes"})
_FALSE = frozenset({"0", "off", "false", "no"})


@dataclass(frozen=True)
class Switch:
    """One switch: its environment variable, default and what it selects."""

    variable: str
    default: bool
    description: str


SWITCHES: dict[str, Switch] = {}
"""Every switch, by environment variable."""


def register(variable: str, default: bool, description: str) -> Switch:
    switch = Switch(variable, default, description)
    SWITCHES[variable] = switch
    return switch


def enabled(variable: str) -> bool:
    """Whether the switch ``variable`` (registered) is on."""
    switch = SWITCHES[variable]
    raw = os.environ.get(variable, "").strip().lower()
    if not raw:
        return switch.default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(
        f"{variable}={os.environ[variable]!r} is not a valid switch value; use 1 or 0 "
        f"({switch.description})"
    )


def report() -> dict[str, bool]:
    """The state of every switch (for benchmark records and bug reports)."""
    return {name: enabled(name) for name in sorted(SWITCHES)}
