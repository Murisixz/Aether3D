"""Brushes: what a dab does to the vertices under it (docs/MODELING.md, "Brushes").

Every brush is a closed-form displacement of the vertices within ``radius`` of the dab
center ``c``, weighted per vertex by

``w = falloff(d / radius) * pressure * (1 - mask) * automask``

(``d``: the Euclidean distance, the distance across the view direction for
``falloff_shape="projected"``, or the distance along the surface for ``geodesic``).
``N`` and ``P`` are the *area normal* and *area center*: the ``w``-weighted mean of the
vertex normals and positions within ``normal_radius * radius``. ``s`` is the strength,
``sign`` the direction (``+1``, or ``-1`` to invert) and ``r`` the radius. All lengths
are in world units, so a brush acts the same way on any object transform.

=============  =================================================================
draw           ``sign * s * r * w * N``: raise or lower along the area normal
inflate        ``sign * s * r * w * n_v``: along each vertex's own normal
clay           toward a plane ``s * r * 0.5`` above the area plane (``sign`` side):
               vertices below it close half the gap per dab (``* w``)
flatten        ``-s * w * h * N`` with ``h = (p - P) . N``: toward the area plane
fill           flatten, only vertices below the plane
scrape         flatten, only vertices above the plane
pinch          ``s * w * T(c - p)``: toward the center within the tangent plane (``T``
               removes the ``N`` component); ``s`` is the fraction per dab (<= 1)
crease         pinch at half strength plus draw at half strength (``sign = -1`` by
               default: a groove)
layer          raise to a fixed height ``sign * s * r`` above the position at the
               stroke's start, along the normal there, never beyond it
grab           ``p = p0 + w0 * (c - c0)``: the vertices under the first dab follow the
               stroke (``w0`` and ``p0`` taken once)
elastic        a regularized Kelvinlet (de Goes and James 2017): ``p = p0 + u(p0 - c0)``,
               the elastic response to the stroke's displacement, ``u(0) = c - c0``,
               with regularization ``eps = r`` and Poisson ratio ``poisson_ratio``
snake_hook     grab re-taken at every dab: pulls geometry along the path
twist          rotation about ``N`` through ``c`` by ``sign * s * w * 90`` degrees
smooth         ``s * w * L(p)``: ``L`` the umbrella Laplacian (``"laplacian"``), a
               Taubin lambda/mu pair (``"taubin"``, no shrinkage) or its tangential part
               (``"relax"``: evens the vertices out without changing the shape);
               boundary vertices follow their boundary neighbours
mask           ``mask += sign * s * w`` (clamped to 0..1)
=============  =================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import ValidationError

KINDS = (
    "draw",
    "inflate",
    "clay",
    "flatten",
    "fill",
    "scrape",
    "pinch",
    "crease",
    "layer",
    "grab",
    "elastic",
    "snake_hook",
    "twist",
    "smooth",
    "mask",
)
"""Every brush kind."""

FALLOFFS = ("smooth", "smoother", "sphere", "root", "sharp", "linear", "constant")
"""Built-in falloff curves (a custom curve is a list of ``[x, y]`` points)."""

STATEFUL = {"grab", "elastic", "layer"}
"""Brushes that capture data at the stroke's first dab."""

AUTOMASKS = ("topology", "face_groups", "boundary")
SMOOTH_METHODS = ("laplacian", "taubin", "relax")
PRESSURE_TARGETS = ("strength", "radius", "both", "none")


def falloff_curve(name: Any) -> Any:
    """``f(x)`` for ``x`` in ``[0, 1]`` (distance / radius): 1 at the center, 0 at the edge
    (except ``constant``)."""
    if isinstance(name, str):
        curves = {
            "smooth": lambda x: 1 - (3 * x * x - 2 * x**3),
            "smoother": lambda x: 1 - x**3 * (x * (6 * x - 15) + 10),
            "sphere": lambda x: np.sqrt(np.maximum(1 - x * x, 0.0)),
            "root": lambda x: 1 - np.sqrt(x),
            "sharp": lambda x: (1 - x) ** 2,
            "linear": lambda x: 1 - x,
            "constant": lambda x: np.ones_like(x),
        }
        if name not in curves:
            raise ValidationError(f"falloff must be one of {FALLOFFS} or a list of points")
        return curves[name]
    pts = np.asarray(name, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or len(pts) > 64:
        raise ValidationError("a custom falloff is a list of 2 to 64 [x, y] points")
    if not np.all(np.isfinite(pts)) or pts[0, 0] != 0 or pts[-1, 0] != 1:
        raise ValidationError("a custom falloff's x values run from 0 to 1")
    if np.any(np.diff(pts[:, 0]) <= 0) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] > 1):
        raise ValidationError("custom falloff points need increasing x and y in [0, 1]")
    xs, ys = pts[:, 0].copy(), pts[:, 1].copy()
    return lambda x: np.interp(x, xs, ys)


@dataclass(frozen=True)
class Brush:
    """A brush's settings (validated; see the module documentation)."""

    kind: str
    radius: float
    strength: float = 0.5
    direction: int = 1
    falloff: Any = "smooth"
    hardness: float = 0.0
    spacing: float = 0.1
    normal: str = "area"
    normal_radius: float = 1.0
    falloff_shape: str = "sphere"
    geodesic: bool = False
    front_faces_only: bool = False
    automask: tuple[str, ...] = ()
    use_mask: bool = True
    symmetry: str = ""
    pressure: str = "strength"
    smooth_method: str = "laplacian"
    poisson_ratio: float = 0.4
    dyntopo: dict | None = field(default=None, hash=False, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValidationError(f"unknown brush {self.kind!r} (brushes: {', '.join(KINDS)})")
        _number(self.radius, "radius", 0.0, math.inf, open_low=True)
        _number(self.strength, "strength", 0.0, 10.0)
        if self.direction not in (1, -1) or isinstance(self.direction, bool):
            raise ValidationError("direction must be 1 or -1")
        falloff_curve(self.falloff)
        _number(self.hardness, "hardness", 0.0, 0.99)
        _number(self.spacing, "spacing", 0.01, 10.0)
        if self.normal not in ("area", "vertex", "view"):
            raise ValidationError("normal must be 'area', 'vertex' or 'view'")
        _number(self.normal_radius, "normal_radius", 0.01, 2.0)
        if self.falloff_shape not in ("sphere", "projected"):
            raise ValidationError("falloff_shape must be 'sphere' or 'projected'")
        bad = [a for a in self.automask if a not in AUTOMASKS]
        if bad:
            raise ValidationError(f"unknown automask {bad} (choices: {AUTOMASKS})")
        if any(c not in "xyz" for c in self.symmetry) or len(set(self.symmetry)) != len(
            self.symmetry
        ):
            raise ValidationError("symmetry is a subset of 'xyz'")
        if self.pressure not in PRESSURE_TARGETS:
            raise ValidationError(f"pressure must be one of {PRESSURE_TARGETS}")
        if self.smooth_method not in SMOOTH_METHODS:
            raise ValidationError(f"smooth_method must be one of {SMOOTH_METHODS}")
        _number(self.poisson_ratio, "poisson_ratio", 0.0, 0.499)
        if self.kind in ("pinch", "flatten", "fill", "scrape", "smooth") and self.strength > 1:
            raise ValidationError(f"{self.kind} strength is a fraction per dab (at most 1)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Brush:
        if not isinstance(data, dict):
            raise ValidationError("a brush is an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValidationError(f"unknown brush settings: {', '.join(unknown)}")
        if "kind" not in data or "radius" not in data:
            raise ValidationError("a brush needs a kind and a radius")
        values = dict(data)
        if "automask" in values:
            if not isinstance(values["automask"], (list, tuple)):
                raise ValidationError("automask is a list")
            values["automask"] = tuple(values["automask"])
        if isinstance(values.get("falloff"), list):
            values["falloff"] = tuple(tuple(p) for p in values["falloff"])
        if values.get("dyntopo") is not None and not isinstance(values["dyntopo"], dict):
            raise ValidationError("dyntopo is an object")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in self.__dataclass_fields__}
        out["automask"] = list(self.automask)
        if not isinstance(self.falloff, str):
            out["falloff"] = [list(p) for p in self.falloff]
        return out

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Falloff of normalized distances (with hardness: full strength up to it)."""
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
        if self.hardness > 0:
            x = np.clip((x - self.hardness) / (1.0 - self.hardness), 0.0, 1.0)
        return np.clip(falloff_curve(self.falloff)(x), 0.0, 1.0)


def _number(value: Any, name: str, lo: float, hi: float, open_low: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise ValidationError(f"{name} must be finite")
    if (v <= lo if open_low else v < lo) or v > hi:
        raise ValidationError(f"{name} must be in {'(' if open_low else '['}{lo}, {hi}]")


def kelvinlet(r: np.ndarray, delta: np.ndarray, eps: float, nu: float) -> np.ndarray:
    """Regularized Kelvinlet displacement (de Goes and James 2017) at offsets ``r`` from
    the load, normalized so the displacement at the load is ``delta``.

    ``u(r) = [(a - b) / r_e I + b / r_e^3 r r^T + a eps^2 / (2 r_e^3) I] F`` with
    ``r_e = sqrt(|r|^2 + eps^2)``, ``b = a / (4 (1 - nu))``, and ``F = 2 eps delta /
    (3a - 2b)`` (so ``u(0) = delta``). The shear modulus (``a = 1 / (4 pi mu)``) cancels.
    """
    beta = 1.0 / (4.0 * (1.0 - nu))  # b / a
    rr = (r * r).sum(axis=1)
    re = np.sqrt(rr + eps * eps)
    scale = 2.0 * eps / (3.0 - 2.0 * beta)
    iso = (1.0 - beta) / re + eps * eps / (2.0 * re**3)
    radial = beta / re**3 * (r @ delta)
    return scale * (iso[:, None] * delta[None, :] + radial[:, None] * r)
