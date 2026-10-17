"""Analytic 3D geometry: frames, curves and surfaces of precision models.

Every surface is a map ``S(u, v)`` with a natural normal ``Su x Sv`` (unit), the
parameterizations of ISO 10303-42 (STEP), so that B-reps export without conversion:

==============  ==============================================================  =========
surface         S(u, v)                                                         periodic
==============  ==============================================================  =========
``Plane``       o + u x + v y                                                   --
``Cylinder``    o + r (cos u x + sin u y) + v z                                 u
``Cone``        o + (r + v tan a) (cos u x + sin u y) + v z                     u
``Sphere``      o + r cos v (cos u x + sin u y) + r sin v z                     u
``Torus``       o + (R + r cos v) (cos u x + sin u y) + r sin v z               u, v
``Extrusion``   C(u) + v d                                                      --
``Revolution``  C(v) rotated by u about the axis (o, z)                         u
``Nurbs``       tensor-product (rational) B-spline                              --
==============  ==============================================================  =========

Curves: ``Line3`` (t = distance from its origin), ``Circle3`` and ``Ellipse3`` (t =
angle in their frame), ``Nurbs3`` (a 3D :class:`~aether3d.cad.nurbs.BSplineCurve`).

All evaluation is vectorized over parameter arrays. ``inverse`` returns the parameters of
the closest surface point (closed forms for the elementary surfaces, Newton for the
others). Transforms are rigid motions (rotations, translations, reflections); uniform
scales are accepted where noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import CadError
from .nurbs import BSplineCurve, BSplineSurface

TAU = 2.0 * math.pi


def unit(v, what: str = "direction") -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n == 0.0:
        raise CadError("DEGENERATE_GEOMETRY", f"{what} has zero length")
    return v / n


def perpendicular(z: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to unit ``z`` (deterministic choice)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = a - (a @ z) * z
    return x / np.linalg.norm(x)


@dataclass(frozen=True)
class Frame:
    """A right-handed orthonormal coordinate frame: origin and unit axes."""

    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @classmethod
    def world(cls) -> Frame:
        return cls(np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])

    @classmethod
    def from_axes(cls, origin, z, x=None) -> Frame:
        """A frame with normal ``z`` and first axis ``x`` (projected; chosen if ``None``)."""
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(o)):
            raise CadError("INVALID_VALUE", "a frame's origin must be finite")
        zz = unit(z, "a frame's normal")
        if x is None:
            xx = perpendicular(zz)
        else:
            xv = np.asarray(x, dtype=np.float64).reshape(3)
            xv = xv - (xv @ zz) * zz
            n = float(np.linalg.norm(xv))
            if n <= 1e-12 * max(1.0, float(np.linalg.norm(x))):
                raise CadError("DEGENERATE_GEOMETRY", "a frame's x axis is parallel to its normal")
            xx = xv / n
        yy = np.cross(zz, xx)
        return cls(o, xx, yy / np.linalg.norm(yy), zz)

    @classmethod
    def from_matrix(cls, m) -> Frame:
        m = np.asarray(m, dtype=np.float64)
        return cls.from_axes(m[:3, 3], m[:3, 2], m[:3, 0])

    @property
    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = self.x, self.y, self.z, self.origin
        return m

    @property
    def rotation(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=1)

    def to_world(self, local) -> np.ndarray:
        p = np.asarray(local, dtype=np.float64)
        if p.shape[-1] == 2:
            return self.origin + p[..., :1] * self.x + p[..., 1:2] * self.y
        return self.origin + p[..., :1] * self.x + p[..., 1:2] * self.y + p[..., 2:3] * self.z

    def to_local(self, world) -> np.ndarray:
        d = np.asarray(world, dtype=np.float64) - self.origin
        return d @ self.rotation

    def direction_to_world(self, local) -> np.ndarray:
        return np.asarray(local, dtype=np.float64) @ self.rotation.T

    def transformed(self, m) -> Frame:
        """Under a rigid motion ``m`` (4x4); a reflection keeps the frame right-handed by
        flipping ``y`` (callers track the orientation change)."""
        m = np.asarray(m, dtype=np.float64)
        r, t = m[:3, :3], m[:3, 3]
        o = r @ self.origin + t
        x = r @ self.x
        z = r @ self.z
        return Frame.from_axes(o, z / np.linalg.norm(z), x / np.linalg.norm(x))

    def offset(self, distance: float) -> Frame:
        return Frame(self.origin + distance * self.z, self.x, self.y, self.z)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "origin": [float(v) for v in self.origin],
            "x": [float(v) for v in self.x],
            "z": [float(v) for v in self.z],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Frame:
        return cls.from_axes(d["origin"], d["z"], d.get("x"))

    def same(self, other: Frame, tol: float) -> bool:
        return bool(
            np.linalg.norm(self.origin - other.origin) <= tol
            and np.linalg.norm(self.x - other.x) <= 1e-12
            and np.linalg.norm(self.z - other.z) <= 1e-12
        )


def _cs(u):
    u = np.asarray(u, dtype=np.float64)
    return np.cos(u), np.sin(u)


def _comb(frame: Frame, a, b, c=None) -> np.ndarray:
    """``a x + b y (+ c z)`` for arrays ``a``, ``b``, ``c`` (broadcast)."""
    out = np.multiply.outer(a, frame.x) + np.multiply.outer(b, frame.y)
    if c is not None:
        out = out + np.multiply.outer(c, frame.z)
    return out


def _unwrap_near(angle: np.ndarray, ref: float | None) -> np.ndarray:
    if ref is None:
        return angle
    return angle + TAU * np.round((ref - angle) / TAU)


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Line3:
    origin: np.ndarray
    direction: np.ndarray
    kind: str = field(default="line", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=np.float64).reshape(3))
        object.__setattr__(self, "direction", unit(self.direction, "a line's direction"))

    def point(self, t) -> np.ndarray:
        return self.origin + np.multiply.outer(np.asarray(t, dtype=np.float64), self.direction)

    def derivative(self, t, order: int = 1) -> np.ndarray:
        shape = np.shape(t)
        if order == 1:
            return np.broadcast_to(self.direction, (*shape, 3)).copy()
        return np.zeros((*shape, 3))

    def inverse(self, p, near: float | None = None) -> np.ndarray:
        return (np.asarray(p, dtype=np.float64) - self.origin) @ self.direction

    def transformed(self, m) -> Line3:
        m = np.asarray(m, dtype=np.float64)
        return Line3(m[:3, :3] @ self.origin + m[:3, 3], m[:3, :3] @ self.direction)

    periodic = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "line",
            "origin": self.origin.tolist(),
            "direction": self.direction.tolist(),
        }


@dataclass(frozen=True)
class Circle3:
    frame: Frame
    radius: float
    kind: str = field(default="circle", init=False)
    periodic = True

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise CadError("DEGENERATE_GEOMETRY", "a circle's radius must be positive")

    def point(self, t) -> np.ndarray:
        c, s = _cs(t)
        return self.frame.origin + self.radius * _comb(self.frame, c, s)

    def derivative(self, t, order: int = 1) -> np.ndarray:
        c, s = _cs(t)
        k = order % 4
        a, b = [(c, s), (-s, c), (-c, -s), (s, -c)][k]
        return self.radius * _comb(self.frame, a, b)

    def inverse(self, p, near: float | None = None) -> np.ndarray:
        loc = self.frame.to_local(p)
        ang = np.arctan2(loc[..., 1], loc[..., 0])
        return _unwrap_near(ang, near)

    def transformed(self, m) -> Circle3:
        return Circle3(self.frame.transformed(m), self.radius * _scale_of(m))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "circle", "frame": self.frame.to_dict(), "radius": self.radius}


@dataclass(frozen=True)
class Ellipse3:
    frame: Frame
    major: float
    minor: float
    kind: str = field(default="ellipse", init=False)
    periodic = True

    def __post_init__(self) -> None:
        if not (self.major >= self.minor > 0 and math.isfinite(self.major)):
            raise CadError("DEGENERATE_GEOMETRY", "an ellipse needs major >= minor > 0")

    def point(self, t) -> np.ndarray:
        c, s = _cs(t)
        return self.frame.origin + _comb(self.frame, self.major * c, self.minor * s)

    def derivative(self, t, order: int = 1) -> np.ndarray:
        c, s = _cs(t)
        a, b = [(c, s), (-s, c), (-c, -s), (s, -c)][order % 4]
        return _comb(self.frame, self.major * a, self.minor * b)

    def inverse(self, p, near: float | None = None) -> np.ndarray:
        loc = self.frame.to_local(p)
        t = np.arctan2(loc[..., 1] * self.major, loc[..., 0] * self.minor)
        # Newton on d/dt |E(t) - p|^2 = 0 (the angle guess is exact on the ellipse).
        for _ in range(8):
            c, s = np.cos(t), np.sin(t)
            ex, ey = self.major * c - loc[..., 0], self.minor * s - loc[..., 1]
            dx, dy = -self.major * s, self.minor * c
            f = ex * dx + ey * dy
            df = dx * dx + dy * dy + ex * (-self.major * c) + ey * (-self.minor * s)
            t = t - np.where(df > 0, f / np.where(df > 0, df, 1.0), 0.0)
        return _unwrap_near(t, near)

    def transformed(self, m) -> Ellipse3:
        k = _scale_of(m)
        return Ellipse3(self.frame.transformed(m), self.major * k, self.minor * k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ellipse",
            "frame": self.frame.to_dict(),
            "major": self.major,
            "minor": self.minor,
        }


@dataclass(frozen=True)
class Nurbs3:
    curve: BSplineCurve
    kind: str = field(default="nurbs", init=False)
    periodic = False

    def __post_init__(self) -> None:
        if self.curve.dim != 3:
            raise CadError("INVALID_SPLINE", "a 3D curve needs 3D control points")

    def point(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.curve.evaluate(np.atleast_1d(t)).reshape(*t.shape, 3)

    def derivative(self, t, order: int = 1) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.curve.derivatives(np.atleast_1d(t), order)[order].reshape(*t.shape, 3)

    def inverse(self, p, near: float | None = None) -> np.ndarray:
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        return np.array([self.curve.closest(q)[0] for q in p])

    def transformed(self, m) -> Nurbs3:
        return Nurbs3(self.curve.transformed(m))

    def to_dict(self) -> dict[str, Any]:
        c = self.curve
        return {
            "type": "nurbs",
            "degree": c.degree,
            "knots": c.knots.tolist(),
            "points": c.points.tolist(),
            "weights": None if c.weights is None else c.weights.tolist(),
        }


Curve3 = Line3 | Circle3 | Ellipse3 | Nurbs3


def _scale_of(m) -> float:
    m = np.asarray(m, dtype=np.float64)
    return float(abs(np.linalg.det(m[:3, :3])) ** (1.0 / 3.0))


# --------------------------------------------------------------------------- #
# Surfaces
# --------------------------------------------------------------------------- #


class Surface:
    kind = "surface"
    periodic_u = False
    periodic_v = False

    def point(self, u, v) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def derivatives(self, u, v) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def second(self, u, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Second partial derivatives (Suu, Suv, Svv) -- by default central differences."""
        h = 1e-5
        su_p, sv_p = self.derivatives(u + h, v)
        su_m, sv_m = self.derivatives(u - h, v)
        _, sv_vp = self.derivatives(u, v + h)
        _, sv_vm = self.derivatives(u, v - h)
        return (su_p - su_m) / (2 * h), (sv_p - sv_m) / (2 * h), (sv_vp - sv_vm) / (2 * h)

    def normal(self, u, v) -> np.ndarray:
        su, sv = self.derivatives(u, v)
        n = np.cross(su, sv)
        length = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / np.where(length > 0, length, 1.0)

    def inverse(self, p, near: tuple[float, float] | None = None) -> np.ndarray:
        """(u, v) of the closest point (shape (..., 2)); Newton from a coarse guess."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        out = np.array([self._newton_inverse(q, near) for q in p])
        return out

    def _guess(self, q: np.ndarray) -> tuple[float, float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _newton_inverse(self, q: np.ndarray, near) -> np.ndarray:
        u, v = near if near is not None else self._guess(q)
        for _ in range(40):
            s = self.point(np.array([u]), np.array([v]))[0]
            su, sv = (a[0] for a in self.derivatives(np.array([u]), np.array([v])))
            suu, suv, svv = (a[0] for a in self.second(np.array([u]), np.array([v])))
            r = s - q
            g = np.array([r @ su, r @ sv])
            h = np.array(
                [[su @ su + r @ suu, su @ sv + r @ suv], [su @ sv + r @ suv, sv @ sv + r @ svv]]
            )
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            u, v = u - float(step[0]), v - float(step[1])
            u, v = self._clamp(u, v)
            if abs(step[0]) + abs(step[1]) < 1e-15:
                break
        return np.array([u, v])

    def _clamp(self, u: float, v: float) -> tuple[float, float]:
        return u, v

    def distance(self, p) -> np.ndarray:
        uv = self.inverse(p)
        q = self.point(uv[:, 0], uv[:, 1])
        return np.linalg.norm(q - np.atleast_2d(p), axis=1)


@dataclass(frozen=True, eq=False)
class Plane(Surface):
    frame: Frame
    kind = "plane"

    def point(self, u, v) -> np.ndarray:
        return self.frame.origin + _comb(self.frame, np.asarray(u, float), np.asarray(v, float))

    def derivatives(self, u, v):
        shape = np.broadcast(np.asarray(u), np.asarray(v)).shape
        return (
            np.broadcast_to(self.frame.x, (*shape, 3)).copy(),
            np.broadcast_to(self.frame.y, (*shape, 3)).copy(),
        )

    def second(self, u, v):
        z = np.zeros((*np.broadcast(np.asarray(u), np.asarray(v)).shape, 3))
        return z, z, z

    def normal(self, u, v) -> np.ndarray:
        shape = np.broadcast(np.asarray(u), np.asarray(v)).shape
        return np.broadcast_to(self.frame.z, (*shape, 3)).copy()

    def inverse(self, p, near=None) -> np.ndarray:
        return self.frame.to_local(p)[..., :2]

    def signed_distance(self, p) -> np.ndarray:
        return (np.asarray(p, dtype=np.float64) - self.frame.origin) @ self.frame.z

    def transformed(self, m) -> Plane:
        return Plane(self.frame.transformed(m))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "plane", "frame": self.frame.to_dict()}


@dataclass(frozen=True, eq=False)
class Cylinder(Surface):
    frame: Frame
    radius: float
    kind = "cylinder"
    periodic_u = True

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise CadError("DEGENERATE_GEOMETRY", "a cylinder's radius must be positive")

    def point(self, u, v) -> np.ndarray:
        c, s = _cs(u)
        return self.frame.origin + _comb(
            self.frame, self.radius * c, self.radius * s, np.asarray(v, float)
        )

    def derivatives(self, u, v):
        c, s = _cs(u)
        v = np.asarray(v, float)
        su = _comb(self.frame, -self.radius * s, self.radius * c)
        sv = np.broadcast_to(self.frame.z, su.shape).copy()
        return su, sv

    def second(self, u, v):
        c, s = _cs(u)
        suu = _comb(self.frame, -self.radius * c, -self.radius * s)
        z = np.zeros_like(suu)
        return suu, z, z

    def normal(self, u, v) -> np.ndarray:
        c, s = _cs(u)
        return _comb(self.frame, c, s) + 0.0 * np.asarray(v, float)[..., None]

    def inverse(self, p, near=None) -> np.ndarray:
        loc = self.frame.to_local(p)
        u = np.arctan2(loc[..., 1], loc[..., 0])
        if near is not None:
            u = _unwrap_near(u, near[0])
        return np.stack([u, loc[..., 2]], axis=-1)

    def transformed(self, m) -> Cylinder:
        return Cylinder(self.frame.transformed(m), self.radius * _scale_of(m))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "cylinder", "frame": self.frame.to_dict(), "radius": self.radius}


@dataclass(frozen=True, eq=False)
class Cone(Surface):
    """Radius ``radius`` at v = 0, growing by ``tan(half_angle)`` per unit of v."""

    frame: Frame
    radius: float
    half_angle: float
    kind = "cone"
    periodic_u = True

    def __post_init__(self) -> None:
        if not (0 < abs(self.half_angle) < math.pi / 2):
            raise CadError("DEGENERATE_GEOMETRY", "a cone's half angle must be in (0, 90) degrees")
        if not (math.isfinite(self.radius) and self.radius >= 0):
            raise CadError("DEGENERATE_GEOMETRY", "a cone's radius must be >= 0")

    @property
    def slope(self) -> float:
        return math.tan(self.half_angle)

    @property
    def apex_v(self) -> float:
        return -self.radius / self.slope

    def point(self, u, v) -> np.ndarray:
        c, s = _cs(u)
        v = np.asarray(v, float)
        r = self.radius + v * self.slope
        return self.frame.origin + _comb(self.frame, r * c, r * s, v)

    def derivatives(self, u, v):
        c, s = _cs(u)
        v = np.asarray(v, float)
        r = self.radius + v * self.slope
        su = _comb(self.frame, -r * s, r * c)
        sv = _comb(self.frame, self.slope * c, self.slope * s, np.ones_like(r * c))
        return su, sv

    def second(self, u, v):
        c, s = _cs(u)
        v = np.asarray(v, float)
        r = self.radius + v * self.slope
        suu = _comb(self.frame, -r * c, -r * s)
        suv = _comb(self.frame, -self.slope * s, self.slope * c)
        return suu, suv, np.zeros_like(suu)

    def normal(self, u, v) -> np.ndarray:
        c, s = _cs(u)
        v = np.asarray(v, float)
        r = self.radius + v * self.slope
        # Su x Sv / |.|: outward radial tilted against the slope (sign of r).
        cos_a, sin_a = math.cos(self.half_angle), math.sin(self.half_angle)
        sign = np.sign(r) + (r == 0)
        n = _comb(self.frame, cos_a * c, cos_a * s, -sin_a * np.ones_like(c))
        return n * sign[..., None]

    def inverse(self, p, near=None) -> np.ndarray:
        loc = self.frame.to_local(p)
        u = np.arctan2(loc[..., 1], loc[..., 0])
        rho = np.hypot(loc[..., 0], loc[..., 1])
        # Closest point on the generator line (rho, z) = (radius + v slope, v).
        ca, sa = math.cos(self.half_angle), math.sin(self.half_angle)
        v = ((rho - self.radius) * sa + loc[..., 2] * ca) * ca
        if near is not None:
            u = _unwrap_near(u, near[0])
        return np.stack([u, v], axis=-1)

    def transformed(self, m) -> Cone:
        k = _scale_of(m)
        return Cone(self.frame.transformed(m), self.radius * k, self.half_angle)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cone",
            "frame": self.frame.to_dict(),
            "radius": self.radius,
            "half_angle": self.half_angle,
        }


@dataclass(frozen=True, eq=False)
class Sphere(Surface):
    frame: Frame
    radius: float
    kind = "sphere"
    periodic_u = True

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise CadError("DEGENERATE_GEOMETRY", "a sphere's radius must be positive")

    def point(self, u, v) -> np.ndarray:
        cu, su = _cs(u)
        cv, sv = _cs(v)
        r = self.radius
        return self.frame.origin + _comb(self.frame, r * cv * cu, r * cv * su, r * sv)

    def derivatives(self, u, v):
        cu, su = _cs(u)
        cv, sv = _cs(v)
        r = self.radius
        du = _comb(self.frame, -r * cv * su, r * cv * cu, 0.0 * cv)
        dv = _comb(self.frame, -r * sv * cu, -r * sv * su, r * cv)
        return du, dv

    def second(self, u, v):
        cu, su = _cs(u)
        cv, sv = _cs(v)
        r = self.radius
        duu = _comb(self.frame, -r * cv * cu, -r * cv * su, 0.0 * cv)
        duv = _comb(self.frame, r * sv * su, -r * sv * cu, 0.0 * cv)
        dvv = _comb(self.frame, -r * cv * cu, -r * cv * su, -r * sv)
        return duu, duv, dvv

    def normal(self, u, v) -> np.ndarray:
        cu, su = _cs(u)
        cv, sv = _cs(v)
        return _comb(self.frame, cv * cu, cv * su, sv)

    def inverse(self, p, near=None) -> np.ndarray:
        loc = self.frame.to_local(p)
        u = np.arctan2(loc[..., 1], loc[..., 0])
        v = np.arctan2(loc[..., 2], np.hypot(loc[..., 0], loc[..., 1]))
        if near is not None:
            u = _unwrap_near(u, near[0])
        return np.stack([u, v], axis=-1)

    def transformed(self, m) -> Sphere:
        return Sphere(self.frame.transformed(m), self.radius * _scale_of(m))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "sphere", "frame": self.frame.to_dict(), "radius": self.radius}


@dataclass(frozen=True, eq=False)
class Torus(Surface):
    frame: Frame
    major: float
    minor: float
    kind = "torus"
    periodic_u = True
    periodic_v = True

    def __post_init__(self) -> None:
        if not (self.major > self.minor > 0 and math.isfinite(self.major)):
            raise CadError("DEGENERATE_GEOMETRY", "a torus needs major > minor > 0")

    def point(self, u, v) -> np.ndarray:
        cu, su = _cs(u)
        cv, sv = _cs(v)
        r = self.major + self.minor * cv
        return self.frame.origin + _comb(self.frame, r * cu, r * su, self.minor * sv)

    def derivatives(self, u, v):
        cu, su = _cs(u)
        cv, sv = _cs(v)
        r = self.major + self.minor * cv
        du = _comb(self.frame, -r * su, r * cu, 0.0 * cv)
        dv = _comb(self.frame, -self.minor * sv * cu, -self.minor * sv * su, self.minor * cv)
        return du, dv

    def second(self, u, v):
        cu, su = _cs(u)
        cv, sv = _cs(v)
        r = self.major + self.minor * cv
        m = self.minor
        duu = _comb(self.frame, -r * cu, -r * su, 0.0 * cv)
        duv = _comb(self.frame, m * sv * su, -m * sv * cu, 0.0 * cv)
        dvv = _comb(self.frame, -m * cv * cu, -m * cv * su, -m * sv)
        return duu, duv, dvv

    def normal(self, u, v) -> np.ndarray:
        cu, su = _cs(u)
        cv, sv = _cs(v)
        return _comb(self.frame, cv * cu, cv * su, sv)

    def inverse(self, p, near=None) -> np.ndarray:
        loc = self.frame.to_local(p)
        u = np.arctan2(loc[..., 1], loc[..., 0])
        rho = np.hypot(loc[..., 0], loc[..., 1])
        v = np.arctan2(loc[..., 2], rho - self.major)
        if near is not None:
            u = _unwrap_near(u, near[0])
            v = _unwrap_near(v, near[1])
        return np.stack([u, v], axis=-1)

    def transformed(self, m) -> Torus:
        k = _scale_of(m)
        return Torus(self.frame.transformed(m), self.major * k, self.minor * k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "torus",
            "frame": self.frame.to_dict(),
            "major": self.major,
            "minor": self.minor,
        }


@dataclass(frozen=True, eq=False)
class Extrusion(Surface):
    """``curve(u) + v * direction``: a spline profile swept along a straight line."""

    curve: Nurbs3
    direction: np.ndarray
    kind = "extrusion"

    def point(self, u, v) -> np.ndarray:
        return self.curve.point(u) + np.multiply.outer(np.asarray(v, float), self.direction)

    def derivatives(self, u, v):
        du = self.curve.derivative(u, 1)
        dv = np.broadcast_to(self.direction, du.shape).copy()
        return du, dv

    def second(self, u, v):
        duu = self.curve.derivative(u, 2)
        z = np.zeros_like(duu)
        return duu, z, z

    def _guess(self, q):
        lo, hi = self.curve.curve.domain
        us = np.linspace(lo, hi, 8 * len(self.curve.curve.points) + 1)
        c = self.curve.point(us)
        dd = float(self.direction @ self.direction)
        vs = (q - c) @ self.direction / dd
        dist = np.sum((c + np.multiply.outer(vs, self.direction) - q) ** 2, axis=1)
        k = int(np.argmin(dist))
        return float(us[k]), float(vs[k])

    def _clamp(self, u, v):
        lo, hi = self.curve.curve.domain
        return min(max(u, lo), hi), v

    def transformed(self, m) -> Extrusion:
        m = np.asarray(m, dtype=np.float64)
        return Extrusion(self.curve.transformed(m), m[:3, :3] @ self.direction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "extrusion",
            "curve": self.curve.to_dict(),
            "direction": self.direction.tolist(),
        }


@dataclass(frozen=True, eq=False)
class Revolution(Surface):
    """``curve(v)`` rotated by angle ``u`` about the frame's z axis (through its origin)."""

    frame: Frame
    curve: Nurbs3
    kind = "revolution"
    periodic_u = True

    def _rot(self, u, pts: np.ndarray) -> np.ndarray:
        loc = self.frame.to_local(pts)
        c, s = _cs(u)
        x = loc[..., 0] * c - loc[..., 1] * s
        y = loc[..., 0] * s + loc[..., 1] * c
        return self.frame.to_world(np.stack([x, y, loc[..., 2]], axis=-1))

    def point(self, u, v) -> np.ndarray:
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        return self._rot(u, self.curve.point(v))

    def derivatives(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        p = self.curve.point(v)
        loc = self.frame.to_local(p)
        c, s = _cs(u)
        dx = -loc[..., 0] * s - loc[..., 1] * c
        dy = loc[..., 0] * c - loc[..., 1] * s
        du = self.frame.direction_to_world(np.stack([dx, dy, np.zeros_like(dx)], axis=-1))
        dloc = self.curve.derivative(v, 1) @ self.frame.rotation
        x = dloc[..., 0] * c - dloc[..., 1] * s
        y = dloc[..., 0] * s + dloc[..., 1] * c
        dv = self.frame.direction_to_world(np.stack([x, y, dloc[..., 2]], axis=-1))
        return du, dv

    def _guess(self, q):
        loc = self.frame.to_local(q)
        u = math.atan2(loc[1], loc[0])
        rho = math.hypot(loc[0], loc[1])
        # Closest curve point to (rho, 0, z) in the curve's own half-plane.
        target = self.frame.to_world(np.array([rho, 0.0, loc[2]]))
        v = float(self.curve.inverse(target)[0])
        return u, v

    def _clamp(self, u, v):
        lo, hi = self.curve.curve.domain
        return u, min(max(v, lo), hi)

    def transformed(self, m) -> Revolution:
        return Revolution(self.frame.transformed(m), self.curve.transformed(m))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "revolution", "frame": self.frame.to_dict(), "curve": self.curve.to_dict()}


@dataclass(frozen=True, eq=False)
class NurbsSurface(Surface):
    surface: BSplineSurface
    kind = "nurbs"

    def point(self, u, v) -> np.ndarray:
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        return self.surface.evaluate(u.ravel(), v.ravel()).reshape(*u.shape, 3)

    def derivatives(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        d = self.surface.derivatives(u.ravel(), v.ravel(), 1)
        return d[1, 0].reshape(*u.shape, 3), d[0, 1].reshape(*u.shape, 3)

    def second(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        d = self.surface.derivatives(u.ravel(), v.ravel(), 2)
        return (
            d[2, 0].reshape(*u.shape, 3),
            d[1, 1].reshape(*u.shape, 3),
            d[0, 2].reshape(*u.shape, 3),
        )

    def _guess(self, q):
        u0, u1, v0, v1 = self.surface.domain
        us, vs = np.meshgrid(np.linspace(u0, u1, 17), np.linspace(v0, v1, 17), indexing="ij")
        pts = self.point(us, vs)
        k = int(np.argmin(np.sum((pts - q) ** 2, axis=-1)))
        return float(us.ravel()[k]), float(vs.ravel()[k])

    def _clamp(self, u, v):
        u0, u1, v0, v1 = self.surface.domain
        return min(max(u, u0), u1), min(max(v, v0), v1)

    def transformed(self, m) -> NurbsSurface:
        return NurbsSurface(self.surface.transformed(m))

    def to_dict(self) -> dict[str, Any]:
        s = self.surface
        return {
            "type": "nurbs",
            "degree_u": s.degree_u,
            "degree_v": s.degree_v,
            "knots_u": s.knots_u.tolist(),
            "knots_v": s.knots_v.tolist(),
            "points": s.points.tolist(),
            "weights": None if s.weights is None else s.weights.tolist(),
        }


def curve_from_dict(d: dict) -> Curve3:
    t = d.get("type")
    if t == "line":
        return Line3(d["origin"], d["direction"])
    if t == "circle":
        return Circle3(Frame.from_dict(d["frame"]), float(d["radius"]))
    if t == "ellipse":
        return Ellipse3(Frame.from_dict(d["frame"]), float(d["major"]), float(d["minor"]))
    if t == "nurbs":
        return Nurbs3(BSplineCurve(int(d["degree"]), d["knots"], d["points"], d.get("weights")))
    raise CadError("INVALID_GEOMETRY", f"unknown curve type {t!r}")


def surface_from_dict(d: dict) -> Surface:
    t = d.get("type")
    if t == "plane":
        return Plane(Frame.from_dict(d["frame"]))
    if t == "cylinder":
        return Cylinder(Frame.from_dict(d["frame"]), float(d["radius"]))
    if t == "cone":
        return Cone(Frame.from_dict(d["frame"]), float(d["radius"]), float(d["half_angle"]))
    if t == "sphere":
        return Sphere(Frame.from_dict(d["frame"]), float(d["radius"]))
    if t == "torus":
        return Torus(Frame.from_dict(d["frame"]), float(d["major"]), float(d["minor"]))
    if t == "extrusion":
        return Extrusion(curve_from_dict(d["curve"]), np.asarray(d["direction"], float))
    if t == "revolution":
        return Revolution(Frame.from_dict(d["frame"]), curve_from_dict(d["curve"]))
    if t == "nurbs":
        return NurbsSurface(
            BSplineSurface(
                int(d["degree_u"]),
                int(d["degree_v"]),
                d["knots_u"],
                d["knots_v"],
                d["points"],
                d.get("weights"),
            )
        )
    raise CadError("INVALID_GEOMETRY", f"unknown surface type {t!r}")
