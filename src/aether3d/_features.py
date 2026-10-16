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


MATERIAL_CACHE = register(
    "AETHER3D_MATERIAL_CACHE",
    True,
    "reuse materials resolved for earlier renders of a scene (rendering/materials.py, "
    "MaterialCache); 0 resolves every material again for every render",
)

GPU_PERSISTENT_BUFFERS = register(
    "AETHER3D_GPU_PERSISTENT_BUFFERS",
    True,
    "keep the GPU's per-frame data buffers between renders and write only what changed "
    "(rendering/gpu/resources.py, FrameBuffers); 0 creates fresh buffers for every render",
)

RETAINED_RENDER_WORLD = register(
    "AETHER3D_RETAINED_RENDER_WORLD",
    True,
    "keep what extraction computed between renders and update it from the history's "
    "change journal (rendering/retained.py, RenderWorld); 0 walks the whole scene for "
    "every render",
)

NATIVE = register(
    "AETHER3D_NATIVE",
    True,
    "use the native kernels of aether3d-native when it is installed (aether3d/_native.py: "
    "ray traversal, transform hierarchies, PNG filters); 0 uses the NumPy implementations",
)

EDITING_BULK = register(
    "AETHER3D_EDITING_BULK",
    True,
    "build an EditableMesh's faces, edges and corners from a Mesh in bulk with NumPy "
    "(editing/core.py, MeshCore._bulk_faces); 0 adds one face at a time",
)
