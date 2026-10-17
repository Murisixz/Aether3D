"""The tolerance policy of precision modeling: every geometric tolerance comes from here.

Tolerances are derived from the size of the model and the unit roundoff of float64
(``u = 2**-53``), never chosen ad hoc at the place they are used:

* ``scale`` (metres) -- the largest coordinate magnitude or extent the computation
  involves, at least :data:`MIN_SCALE` (1 mm), so that small parts are not given
  tolerances below what their own numbers can resolve.
* ``resolution`` -- the distance below which two points are the same point (sketch
  coincidence, B-rep vertex merging, curve-on-surface checks):
  ``max(RESOLUTION_FLOOR, 2**10 * u * scale)`` -- 0.1 nm for anything up to 450 m, and
  about 0.45 um for coordinates near 4,000 km (georeferenced sites): a thousand times the
  spacing of float64 numbers there, so rounding in a few dozen operations never separates
  equal points.
* ``solve`` -- the residual a constraint solver must reach (max norm of length-scaled
  equations): ``max(SOLVE_FLOOR, 2**6 * u * scale)``; well below ``resolution``, so a
  solved coincidence is a coincidence.
* ``angular`` -- the sine of the angle below which two directions are parallel:
  ``ANGULAR`` (1e-12 rad, the value industrial kernels use) or ``resolution / scale``,
  whichever is larger.
* ``rank`` -- relative singular-value threshold for numerical rank (degrees of freedom,
  redundancy): ``1e3 * u * max(m, n)`` times the largest singular value.

Display tessellation has its own explicit tolerances (``TessellationTolerance``: chordal
deviation and angle), set by the caller; they never change the design geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import CadError

UNIT_ROUNDOFF = 2.0**-53
MIN_SCALE = 1e-3
MAX_SCALE = 1e8  # 100,000 km: anything larger is not a building or a part
RESOLUTION_FLOOR = 1e-10
SOLVE_FLOOR = 1e-14
ANGULAR = 1e-12
MAX_COORDINATE = 1e8
"""Coordinates and lengths beyond 100,000 km are refused (``VALUE_TOO_LARGE``): beyond
them float64 cannot hold 0.1 mm, and such numbers in input are mistakes or attacks."""


@dataclass(frozen=True)
class Tolerance:
    """Tolerances for one computation at a given ``scale`` (see the module docstring)."""

    scale: float
    resolution: float
    solve: float
    angular: float

    @classmethod
    def for_scale(cls, scale: float) -> Tolerance:
        if not math.isfinite(scale):
            raise CadError("VALUE_TOO_LARGE", "the model's size is not finite")
        s = min(max(abs(float(scale)), MIN_SCALE), MAX_SCALE)
        resolution = max(RESOLUTION_FLOOR, 2.0**10 * UNIT_ROUNDOFF * s)
        solve = max(SOLVE_FLOOR, 2.0**6 * UNIT_ROUNDOFF * s)
        angular = max(ANGULAR, resolution / s)
        return cls(s, resolution, solve, angular)

    def to_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "resolution": self.resolution,
            "solve": self.solve,
            "angular": self.angular,
        }


def rank_threshold(sigma_max: float, rows: int, cols: int) -> float:
    """Singular values below this are zero (numerical rank)."""
    return sigma_max * 1e3 * UNIT_ROUNDOFF * max(rows, cols, 1)


@dataclass(frozen=True)
class TessellationTolerance:
    """How closely display or analysis meshes follow exact geometry.

    ``chord`` -- the largest distance (metres) between a curve or surface and its
    piecewise-linear approximation; ``angle`` -- the largest angle (radians) between the
    normals or tangents at the two ends of one segment; ``relative`` -- if set, ``chord``
    is ``relative`` times the size of the body instead (whichever is smaller wins when
    both are given); ``max_edge`` -- optional upper bound on segment length.
    """

    chord: float = 1e-3
    angle: float = math.radians(15.0)
    relative: float | None = 2e-3
    max_edge: float | None = None

    def __post_init__(self) -> None:
        for name in ("chord", "angle"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise CadError("INVALID_TOLERANCE", f"{name} must be a positive number")
        if self.angle > math.radians(60):
            raise CadError("INVALID_TOLERANCE", "angle must be at most 60 degrees")
        if self.relative is not None and not (
            isinstance(self.relative, (int, float)) and 1e-7 <= self.relative <= 0.1
        ):
            raise CadError("INVALID_TOLERANCE", "relative must be within 1e-7..0.1")
        if self.max_edge is not None and not (
            isinstance(self.max_edge, (int, float))
            and math.isfinite(self.max_edge)
            and self.max_edge > 0
        ):
            raise CadError("INVALID_TOLERANCE", "max_edge must be a positive number")

    def chord_for(self, size: float) -> float:
        """The chordal deviation to use for a body of this size (metres)."""
        chord = float(self.chord)
        if self.relative is not None and size > 0:
            chord = min(chord, float(self.relative) * size)
        return max(chord, 1e-9)

    def to_dict(self) -> dict:
        return {
            "chord": self.chord,
            "angle_deg": math.degrees(self.angle),
            "relative": self.relative,
            "max_edge": self.max_edge,
        }


DISPLAY = TessellationTolerance()
"""Default display tessellation: 1 mm or 0.2 % of the body, 15 degrees."""

ANALYSIS = TessellationTolerance(chord=1e-6, angle=math.radians(3.0), relative=1e-5)
"""Tessellation for booleans and tessellation-based checks: about 1e-5 of the body."""


def check_coordinate(value: float, what: str) -> float:
    """A finite coordinate or length within :data:`MAX_COORDINATE`."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CadError("INVALID_VALUE", f"must be a number, got {value!r}", what)
    v = float(value)
    if not math.isfinite(v):
        raise CadError("INVALID_VALUE", f"must be finite, got {v}", what)
    if abs(v) > MAX_COORDINATE:
        raise CadError("VALUE_TOO_LARGE", f"{what} = {v:g} m exceeds {MAX_COORDINATE:g} m", what)
    return v
