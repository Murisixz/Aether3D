"""B-spline and NURBS curves and surfaces (vectorized over parameters).

Conventions follow Piegl & Tiller, *The NURBS Book* (2nd ed.): a degree-``p`` curve with
``n + 1`` control points has a knot vector of ``n + p + 2`` non-decreasing knots; the
curve is defined on ``[U[p], U[n + 1]]``. Rational curves carry positive weights and are
evaluated in homogeneous coordinates (A4.2 for derivatives). Supported:

* evaluation and derivatives of any order up to the degree (A2.1 ``FindSpan``, A2.3 basis
  functions and derivatives), vectorized over arrays of parameters;
* knot insertion (A5.1), exact: the curve does not change;
* exact conic arcs as rational quadratics (A7.1: circular arcs up to 360 degrees with
  weights ``cos(theta/2)``);
* global interpolation of points (A9.1, chord-length parameters, averaged knots), used
  to fit traced intersection curves;
* tensor-product surfaces with first and second partial derivatives and normals.

Limits (``MAX_DEGREE``, ``MAX_CONTROL_POINTS``) guard against pathological input: a curve
is data that may come from a file or an agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import CadError

MAX_DEGREE = 9
MAX_CONTROL_POINTS = 10_000


def _check_knots(knots: np.ndarray, degree: int, count: int, what: str) -> None:
    if knots.ndim != 1 or len(knots) != count + degree + 1:
        raise CadError(
            "INVALID_SPLINE",
            f"{what}: a degree-{degree} spline with {count} control points needs "
            f"{count + degree + 1} knots, got {len(knots)}",
        )
    if not np.all(np.isfinite(knots)):
        raise CadError("INVALID_SPLINE", f"{what}: knots must be finite")
    if np.any(np.diff(knots) < 0):
        raise CadError("INVALID_SPLINE", f"{what}: knots must not decrease")
    if knots[count] <= knots[degree]:
        raise CadError("INVALID_SPLINE", f"{what}: the knot vector's domain is empty")
    # Interior multiplicity at most the degree (keeps the curve continuous).
    inner = knots[degree + 1 : count]
    if len(inner):
        _, counts = np.unique(inner, return_counts=True)
        if counts.max() > degree:
            raise CadError(
                "INVALID_SPLINE", f"{what}: an interior knot repeats more than the degree"
            )


def find_spans(knots: np.ndarray, degree: int, n: int, t: np.ndarray) -> np.ndarray:
    """Knot span index of each parameter (A2.1), clamped to the domain ``[U[p], U[n+1]]``."""
    lo, hi = knots[degree], knots[n + 1]
    t = np.clip(t, lo, hi)
    spans = np.searchsorted(knots, t, side="right") - 1
    return np.clip(spans, degree, n)


def basis_derivatives(
    knots: np.ndarray, degree: int, spans: np.ndarray, t: np.ndarray, order: int
) -> np.ndarray:
    """Non-zero basis functions and derivatives (A2.3), shape ``(len(t), order + 1, p + 1)``.

    ``out[k, d, j]`` is the ``d``-th derivative of ``N_{span-p+j, p}`` at ``t[k]``.
    """
    p = degree
    m = len(t)
    order = min(order, p)
    ndu = np.zeros((m, p + 1, p + 1))
    ndu[:, 0, 0] = 1.0
    left = np.zeros((m, p + 1))
    right = np.zeros((m, p + 1))
    for j in range(1, p + 1):
        left[:, j] = t - knots[spans + 1 - j]
        right[:, j] = knots[spans + j] - t
        saved = np.zeros(m)
        for r in range(j):
            # Lower triangle: knot differences.
            ndu[:, j, r] = right[:, r + 1] + left[:, j - r]
            with np.errstate(divide="ignore", invalid="ignore"):
                temp = np.where(ndu[:, j, r] != 0, ndu[:, r, j - 1] / ndu[:, j, r], 0.0)
            ndu[:, r, j] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        ndu[:, j, j] = saved
    out = np.zeros((m, order + 1, p + 1))
    out[:, 0, :] = ndu[:, :, p]
    if order == 0:
        return out
    for r in range(p + 1):
        a = np.zeros((m, 2, p + 1))
        a[:, 0, 0] = 1.0
        s1, s2 = 0, 1
        for k in range(1, order + 1):
            d = np.zeros(m)
            rk, pk = r - k, p - k
            with np.errstate(divide="ignore", invalid="ignore"):
                if r >= k:
                    a[:, s2, 0] = np.where(
                        ndu[:, pk + 1, rk] != 0, a[:, s1, 0] / ndu[:, pk + 1, rk], 0.0
                    )
                    d = a[:, s2, 0] * ndu[:, rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r
                for j in range(j1, j2 + 1):
                    a[:, s2, j] = np.where(
                        ndu[:, pk + 1, rk + j] != 0,
                        (a[:, s1, j] - a[:, s1, j - 1]) / ndu[:, pk + 1, rk + j],
                        0.0,
                    )
                    d = d + a[:, s2, j] * ndu[:, rk + j, pk]
                if r <= pk:
                    a[:, s2, k] = np.where(
                        ndu[:, pk + 1, r] != 0, -a[:, s1, k - 1] / ndu[:, pk + 1, r], 0.0
                    )
                    d = d + a[:, s2, k] * ndu[:, r, pk]
            out[:, k, r] = d
            s1, s2 = s2, s1
    factor = float(p)
    for k in range(1, order + 1):
        out[:, k, :] *= factor
        factor *= p - k
    return out


@dataclass(frozen=True)
class BSplineCurve:
    """A (rational) B-spline curve in 2 or 3 dimensions; immutable (arrays read-only)."""

    degree: int
    knots: np.ndarray
    points: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.degree, (int, np.integer)) or not 1 <= self.degree <= MAX_DEGREE:
            raise CadError("INVALID_SPLINE", f"degree must be 1..{MAX_DEGREE}")
        pts = np.array(self.points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise CadError("INVALID_SPLINE", "control points must be an (n, 2) or (n, 3) array")
        if not len(pts) > self.degree:
            raise CadError(
                "INVALID_SPLINE",
                f"a degree-{self.degree} curve needs at least {self.degree + 1} points",
            )
        if len(pts) > MAX_CONTROL_POINTS:
            raise CadError("INVALID_SPLINE", f"more than {MAX_CONTROL_POINTS} control points")
        if not np.all(np.isfinite(pts)):
            raise CadError("INVALID_SPLINE", "control points must be finite")
        knots = np.array(self.knots, dtype=np.float64)
        _check_knots(knots, int(self.degree), len(pts), "curve")
        w = None
        if self.weights is not None:
            w = np.array(self.weights, dtype=np.float64)
            if w.shape != (len(pts),) or not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise CadError("INVALID_SPLINE", "weights must be positive, one per point")
            if np.all(w == w[0]):
                w = None if w[0] == 1.0 else w
        for a in (pts, knots, w):
            if a is not None:
                a.setflags(write=False)
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "weights", w)

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def domain(self) -> tuple[float, float]:
        n = len(self.points) - 1
        return float(self.knots[self.degree]), float(self.knots[n + 1])

    @property
    def rational(self) -> bool:
        return self.weights is not None

    def derivatives(self, t: np.ndarray | float, order: int = 0) -> np.ndarray:
        """``out[d, k]`` = ``d``-th derivative at ``t[k]`` (shape ``(order+1, len(t), dim)``)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        p, n = self.degree, len(self.points) - 1
        spans = find_spans(self.knots, p, n, t)
        lo, hi = self.domain
        tc = np.clip(t, lo, hi)
        basis = basis_derivatives(self.knots, p, spans, tc, order)
        idx = spans[:, None] - p + np.arange(p + 1)[None, :]  # (m, p+1)
        if self.weights is None:
            ctrl = self.points[idx]  # (m, p+1, dim)
            out = np.einsum("mdj,mjc->dmc", basis, ctrl)
            if order > p:
                out = np.concatenate([out, np.zeros((order - p, len(t), self.dim))])
            return out
        w = self.weights[idx]
        hom = np.concatenate([self.points[idx] * w[..., None], w[..., None]], axis=2)
        a = np.einsum("mdj,mjc->dmc", basis, hom)  # (order'+1, m, dim+1)
        if order > p:
            a = np.concatenate([a, np.zeros((order - p, len(t), self.dim + 1))])
        aw, wd = a[..., :-1], a[..., -1]
        out = np.zeros_like(aw)
        for k in range(order + 1):  # A4.2
            v = aw[k].copy()
            for i in range(1, k + 1):
                v -= math.comb(k, i) * wd[i][:, None] * out[k - i]
            out[k] = v / wd[0][:, None]
        return out

    def evaluate(self, t: np.ndarray | float) -> np.ndarray:
        return self.derivatives(t, 0)[0]

    def insert_knot(self, u: float, times: int = 1) -> BSplineCurve:
        """The same curve with ``u`` inserted ``times`` times (A5.1)."""
        p, n = self.degree, len(self.points) - 1
        lo, hi = self.domain
        if not lo < u < hi:
            raise CadError("INVALID_SPLINE", "the knot must be inside the domain")
        s = int(np.sum(self.knots == u))
        if times < 1 or s + times > p:
            raise CadError("INVALID_SPLINE", "knot multiplicity would exceed the degree")
        w = self.weights if self.weights is not None else np.ones(n + 1)
        pw = np.concatenate([self.points * w[:, None], w[:, None]], axis=1)
        knots = self.knots
        for _ in range(times):
            k = int(find_spans(knots, p, len(pw) - 1, np.array([u]))[0])
            s = int(np.sum(knots == u))
            new = np.zeros((len(pw) + 1, pw.shape[1]))
            new[: k - p + 1] = pw[: k - p + 1]
            new[k - s + 1 :] = pw[k - s :]
            for i in range(k - p + 1, k - s + 1):
                alpha = (u - knots[i]) / (knots[i + p] - knots[i])
                new[i] = alpha * pw[i] + (1 - alpha) * pw[i - 1]
            knots = np.concatenate([knots[: k + 1], [u], knots[k + 1 :]])
            pw = new
        weights = pw[:, -1]
        return BSplineCurve(p, knots, pw[:, :-1] / weights[:, None], weights)

    def transformed(self, matrix: np.ndarray) -> BSplineCurve:
        """Under an affine map (NURBS are affinely invariant): 3D curves, 4x4 matrix."""
        m = np.asarray(matrix, dtype=np.float64)
        pts = self.points @ m[:3, :3].T + m[:3, 3]
        return BSplineCurve(self.degree, self.knots, pts, self.weights)

    def sample(self, count: int) -> np.ndarray:
        lo, hi = self.domain
        return self.evaluate(np.linspace(lo, hi, count))

    def closest(self, point: np.ndarray, samples: int = 64) -> tuple[float, np.ndarray]:
        """Parameter and point of the curve nearest ``point`` (sampling, then Newton)."""
        lo, hi = self.domain
        n = max(samples, 8 * (len(self.points) - self.degree))
        ts = np.linspace(lo, hi, n)
        pts = self.evaluate(ts)
        point = np.asarray(point, dtype=np.float64)
        t = float(ts[int(np.argmin(np.sum((pts - point) ** 2, axis=1)))])
        for _ in range(30):
            d = self.derivatives(np.array([t]), 2)
            c, c1, c2 = d[0, 0], d[1, 0], d[2, 0]
            diff = c - point
            f = float(diff @ c1)
            df = float(c1 @ c1 + diff @ c2)
            if df <= 0:
                break
            step = f / df
            t_new = min(max(t - step, lo), hi)
            if abs(t_new - t) <= 1e-15 * max(1.0, abs(t)):
                t = t_new
                break
            t = t_new
        return t, self.evaluate(np.array([t]))[0]


def circle_arc_nurbs(
    center: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    radius: float,
    start: float,
    sweep: float,
) -> BSplineCurve:
    """An exact circular arc as a rational quadratic B-spline (A7.1).

    Angles in radians; ``sweep`` in (0, 2 pi]; the arc runs counter-clockwise about
    ``x_axis x y_axis`` from ``start``.
    """
    if not 0 < abs(sweep) <= 2 * math.pi + 1e-12:
        raise CadError("INVALID_SPLINE", "an arc's sweep must be in (0, 360] degrees")
    narcs = max(1, math.ceil(abs(sweep) / (math.pi / 2) - 1e-9))
    dtheta = sweep / narcs
    w1 = math.cos(dtheta / 2)
    center = np.asarray(center, float)
    x_axis, y_axis = np.asarray(x_axis, float), np.asarray(y_axis, float)

    def at(a: float) -> np.ndarray:
        return center + radius * (math.cos(a) * x_axis + math.sin(a) * y_axis)

    pts = [at(start)]
    weights = [1.0]
    angle = start
    for _ in range(narcs):
        a0, a1 = angle, angle + dtheta
        p2 = at(a1)
        # Middle control point: intersection of the end tangents.
        mid_angle = (a0 + a1) / 2
        dist = radius / math.cos(dtheta / 2)
        p1 = center + dist * (math.cos(mid_angle) * x_axis + math.sin(mid_angle) * y_axis)
        pts += [p1, p2]
        weights += [w1, 1.0]
        angle = a1
    knots = [0.0, 0.0, 0.0]
    for i in range(1, narcs):
        knots += [i / narcs, i / narcs]
    knots += [1.0, 1.0, 1.0]
    return BSplineCurve(2, np.array(knots), np.array(pts), np.array(weights))


def interpolate(points: np.ndarray, degree: int = 3) -> BSplineCurve:
    """The B-spline of ``degree`` through ``points`` (A9.1: chord-length parameters,
    averaged knots); a polyline for two points."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts) - 1
    if n < 1:
        raise CadError("INVALID_SPLINE", "interpolation needs at least two points")
    p = min(degree, n)
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(chords.sum())
    if total <= 0:
        raise CadError("INVALID_SPLINE", "interpolation points coincide")
    params = np.concatenate([[0.0], np.cumsum(chords) / total])
    params[-1] = 1.0
    knots = np.zeros(n + p + 2)
    knots[-(p + 1) :] = 1.0
    for j in range(1, n - p + 1):
        knots[j + p] = params[j : j + p].mean()
    spans = find_spans(knots, p, n, params)
    basis = basis_derivatives(knots, p, spans, params, 0)[:, 0, :]
    a = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        a[k, spans[k] - p : spans[k] + 1] = basis[k]
    ctrl = np.linalg.solve(a, pts)
    return BSplineCurve(p, knots, ctrl)


@dataclass(frozen=True)
class BSplineSurface:
    """A (rational) tensor-product B-spline surface; ``points`` has shape (nu, nv, 3)."""

    degree_u: int
    degree_v: int
    knots_u: np.ndarray
    knots_v: np.ndarray
    points: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        pts = np.array(self.points, dtype=np.float64)
        if pts.ndim != 3 or pts.shape[2] != 3:
            raise CadError("INVALID_SPLINE", "surface control points must be (nu, nv, 3)")
        for name in ("degree_u", "degree_v"):
            d = getattr(self, name)
            if not isinstance(d, (int, np.integer)) or not 1 <= d <= MAX_DEGREE:
                raise CadError("INVALID_SPLINE", f"{name} must be 1..{MAX_DEGREE}")
        if pts.shape[0] <= self.degree_u or pts.shape[1] <= self.degree_v:
            raise CadError("INVALID_SPLINE", "too few control points for the degrees")
        if pts.shape[0] * pts.shape[1] > MAX_CONTROL_POINTS:
            raise CadError("INVALID_SPLINE", f"more than {MAX_CONTROL_POINTS} control points")
        if not np.all(np.isfinite(pts)):
            raise CadError("INVALID_SPLINE", "control points must be finite")
        ku = np.array(self.knots_u, dtype=np.float64)
        kv = np.array(self.knots_v, dtype=np.float64)
        _check_knots(ku, int(self.degree_u), pts.shape[0], "surface (u)")
        _check_knots(kv, int(self.degree_v), pts.shape[1], "surface (v)")
        w = None
        if self.weights is not None:
            w = np.array(self.weights, dtype=np.float64)
            if w.shape != pts.shape[:2] or not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise CadError("INVALID_SPLINE", "weights must be positive, one per point")
            if np.all(w == 1.0):
                w = None
        for a in (pts, ku, kv, w):
            if a is not None:
                a.setflags(write=False)
        object.__setattr__(self, "degree_u", int(self.degree_u))
        object.__setattr__(self, "degree_v", int(self.degree_v))
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "knots_u", ku)
        object.__setattr__(self, "knots_v", kv)
        object.__setattr__(self, "weights", w)

    @property
    def domain(self) -> tuple[float, float, float, float]:
        nu, nv = self.points.shape[:2]
        return (
            float(self.knots_u[self.degree_u]),
            float(self.knots_u[nu]),
            float(self.knots_v[self.degree_v]),
            float(self.knots_v[nv]),
        )

    def derivatives(self, u: np.ndarray, v: np.ndarray, order: int = 1) -> np.ndarray:
        """``out[k, l, m]`` = d^(k+l) S / du^k dv^l at sample ``m``; shape (o+1, o+1, m, 3)."""
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        p, q = self.degree_u, self.degree_v
        nu, nv = self.points.shape[:2]
        u0, u1, v0, v1 = self.domain
        uc, vc = np.clip(u, u0, u1), np.clip(v, v0, v1)
        su = find_spans(self.knots_u, p, nu - 1, uc)
        sv = find_spans(self.knots_v, q, nv - 1, vc)
        bu = basis_derivatives(self.knots_u, p, su, uc, order)  # (m, du+1, p+1)
        bv = basis_derivatives(self.knots_v, q, sv, vc, order)
        iu = su[:, None] - p + np.arange(p + 1)[None, :]
        iv = sv[:, None] - q + np.arange(q + 1)[None, :]
        ctrl = self.points[iu[:, :, None], iv[:, None, :]]  # (m, p+1, q+1, 3)
        du, dv = bu.shape[1] - 1, bv.shape[1] - 1
        if self.weights is None:
            s = np.einsum("mkj,mli,mjic->klmc", bu, bv, ctrl)
            out = np.zeros((order + 1, order + 1, len(u), 3))
            out[: du + 1, : dv + 1] = s
            return out
        w = self.weights[iu[:, :, None], iv[:, None, :]]
        hom = np.concatenate([ctrl * w[..., None], w[..., None]], axis=3)
        a = np.zeros((order + 1, order + 1, len(u), 4))
        a[: du + 1, : dv + 1] = np.einsum("mkj,mli,mjic->klmc", bu, bv, hom)
        aw, wd = a[..., :3], a[..., 3]
        out = np.zeros_like(aw)
        for k in range(order + 1):  # A4.4
            for ell in range(order + 1 - k):
                val = aw[k, ell].copy()
                for j in range(1, ell + 1):
                    val -= math.comb(ell, j) * wd[0, j][:, None] * out[k, ell - j]
                for i in range(1, k + 1):
                    val -= math.comb(k, i) * wd[i, 0][:, None] * out[k - i, ell]
                    v2 = np.zeros_like(val)
                    for j in range(1, ell + 1):
                        v2 += math.comb(ell, j) * wd[i, j][:, None] * out[k - i, ell - j]
                    val -= math.comb(k, i) * v2
                out[k, ell] = val / wd[0, 0][:, None]
        return out

    def evaluate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.derivatives(u, v, 0)[0, 0]

    def transformed(self, matrix: np.ndarray) -> BSplineSurface:
        m = np.asarray(matrix, dtype=np.float64)
        pts = self.points @ m[:3, :3].T + m[:3, 3]
        return BSplineSurface(
            self.degree_u, self.degree_v, self.knots_u, self.knots_v, pts, self.weights
        )
