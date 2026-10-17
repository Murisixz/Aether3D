"""Mass properties of B-reps from their exact surfaces (not from a tessellation).

Volume, centroid and the second moments are volume integrals of polynomials; the
divergence theorem turns each into a sum over faces of ``integral P(S) . N dA`` with the
outward normal ``N dA = Su x Sv du dv`` (the sign comes from the loops' orientation, see
below). Each face integral over its trimmed parameter domain ``D`` is turned into a
boundary integral by Green's theorem::

    integral_D g(u, v) du dv  =  loop integral of G(u, v) dv,   G = integral_u0^u g(s, v) ds

evaluated along the face's own edge curves: Gauss-Legendre nodes on every edge (split
into pieces), the parameters (u, v) of each node from the surface's exact inverse, ``dv``
from the edge's tangent, and ``G`` by Gauss-Legendre in ``u``. Because the loops are
stored counter-clockwise seen from outside, the integral comes out with the outward
orientation whatever the face's ``sense``.

Planar faces bounded by lines and arcs integrate to machine precision; curved faces with
smooth integrands converge exponentially. Each result carries an error estimate: the
difference between this and a run with twice the edge subdivisions. Coordinates are
shifted to the body's center first, so far-from-origin models (georeferenced sites) do
not lose digits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import CadError
from ..geom3d import Circle3, Cone, Ellipse3, Line3, Plane, Sphere
from .topology import Body

_GX, _GW = np.polynomial.legendre.leggauss(16)


@dataclass
class MassProperties:
    volume: float
    area: float
    centroid: np.ndarray
    inertia: np.ndarray  # about the centroid, unit density
    principal_moments: np.ndarray
    principal_axes: np.ndarray
    error_volume: float
    error_area: float
    method: str = "boundary quadrature on exact surfaces"

    def to_dict(self, density: float | None = None) -> dict:
        out = {
            "volume": self.volume,
            "area": self.area,
            "centroid": self.centroid.tolist(),
            "inertia_unit_density": self.inertia.tolist(),
            "principal_moments_unit_density": self.principal_moments.tolist(),
            "principal_axes": self.principal_axes.tolist(),
            "estimated_error": {"volume": self.error_volume, "area": self.error_area},
            "method": self.method,
        }
        if density is not None:
            out["density"] = density
            out["mass"] = density * self.volume
            out["inertia"] = (density * self.inertia).tolist()
        return out


def _edge_nodes(edge, pieces: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes and weights in t over the edge, split into ``pieces``."""
    t0, t1 = edge.t0, edge.t1
    k = pieces
    c = edge.curve
    if isinstance(c, (Circle3, Ellipse3)):
        k = pieces * max(1, int(np.ceil((t1 - t0) / (np.pi / 4))))
    elif not isinstance(c, Line3):
        k = pieces * max(4, len(getattr(c, "curve", c).points) if hasattr(c, "curve") else 4)
    edges = np.linspace(t0, t1, k + 1)
    a, b = edges[:-1, None], edges[1:, None]
    ts = (0.5 * (b - a) * _GX[None] + 0.5 * (a + b)).ravel()
    ws = (0.5 * (b - a) * _GW[None]).ravel()
    return ts, ws


def _integrand(s, su_: np.ndarray, sv_: np.ndarray, center: np.ndarray) -> np.ndarray:
    """The 11 integrands at surface parameters (flat arrays): P(S) . (Su x Sv) and |Su x Sv|."""
    pts = s.point(su_, sv_) - center
    dsu, dsv = s.derivatives(su_, sv_)
    n = np.cross(dsu, dsv)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack(
        [
            x * n[:, 0],
            0.5 * x * x * n[:, 0],
            0.5 * y * y * n[:, 1],
            0.5 * z * z * n[:, 2],
            x**3 / 3.0 * n[:, 0],
            y**3 / 3.0 * n[:, 1],
            z**3 / 3.0 * n[:, 2],
            0.5 * x * x * y * n[:, 0],
            0.5 * x * x * z * n[:, 0],
            0.5 * y * y * z * n[:, 1],
            np.linalg.norm(n, axis=1),
        ],
        axis=1,
    )


_TURN_SAMPLES = 96


def _full_turn(s, v: np.ndarray, center: np.ndarray) -> np.ndarray:
    """G_full(v) = integral over a whole turn of u of the integrands (periodic in u, so
    the trapezoidal rule converges exponentially)."""
    u = np.linspace(0.0, 2 * np.pi, _TURN_SAMPLES, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    g = _integrand(s, uu.ravel(), vv.ravel(), center).reshape(len(u), len(v), 11)
    return g.sum(axis=0) * (2 * np.pi / _TURN_SAMPLES)


def _turn_integral(s, v0: float, v1: float, center: np.ndarray) -> np.ndarray:
    """H(v1) - H(v0): the integral over v in [v0, v1] of G_full."""
    if v0 == v1:
        return np.zeros(11)
    k = 16
    edges = np.linspace(v0, v1, k + 1)
    a, b = edges[:-1, None], edges[1:, None]
    vs = (0.5 * (b - a) * _GX[None] + 0.5 * (a + b)).ravel()
    ws = (0.5 * (b - a) * _GW[None]).ravel()
    return np.einsum("vk,v->k", _full_turn(s, vs, center), ws)


def _pole_v(s, above: bool) -> float:
    if isinstance(s, Sphere):
        return np.pi / 2 if above else -np.pi / 2
    if isinstance(s, Cone):
        apex = s.apex_v
        return apex
    raise CadError(
        "UNSUPPORTED_GEOMETRY",
        f"a face wrapping once around a {s.kind} without a second boundary",
    )


def _face_integrals(body: Body, face, center: np.ndarray, pieces: int) -> np.ndarray:
    """[V, Mx, My, Mz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, Area] contributions of one face."""
    s = face.surface
    # Boundary nodes along every loop, in loop order.
    nodes_p, nodes_d, nodes_w = [], [], []
    loop_slices = []
    for loop in face.loops:
        start = sum(len(x) for x in nodes_p)
        for ei, fwd in loop:
            e = body.edges[ei]
            ts, ws = _edge_nodes(e, pieces)
            p = e.curve.point(ts)
            d = e.curve.derivative(ts, 1)
            if not fwd:
                p, d, ws = p[::-1], -d[::-1], ws[::-1]
            nodes_p.append(p)
            nodes_d.append(d)
            nodes_w.append(ws)
        loop_slices.append(slice(start, sum(len(x) for x in nodes_p)))
    p = np.concatenate(nodes_p)
    d = np.concatenate(nodes_d)
    w = np.concatenate(nodes_w)
    uv = face.to_window(s.inverse(p))
    su, sv = s.derivatives(uv[:, 0], uv[:, 1])
    # d(uv)/dt from C'(t) = Su du/dt + Sv dv/dt (least squares on the tangent plane).
    a11 = np.sum(su * su, axis=1)
    a12 = np.sum(su * sv, axis=1)
    a22 = np.sum(sv * sv, axis=1)
    b1 = np.sum(su * d, axis=1)
    b2 = np.sum(sv * d, axis=1)
    det = a11 * a22 - a12 * a12
    ok = det > 1e-300
    dv = np.where(ok, (a11 * b2 - a12 * b1) / np.where(ok, det, 1.0), 0.0)
    u_ref = 0.5 * (float(uv[:, 0].min()) + float(uv[:, 0].max()))
    # Inner integral over s in [u_ref, u] at each node's v.
    half = 0.5 * (uv[:, 0] - u_ref)
    s_nodes = u_ref + half[:, None] * (_GX[None] + 1.0)  # (B, n)
    v_nodes = np.broadcast_to(uv[:, 1:2], s_nodes.shape)
    g = _integrand(s, s_nodes.ravel(), v_nodes.ravel(), center).reshape(len(uv), len(_GX), 11)
    big_g = np.einsum("bnk,n->bk", g, _GW) * half[:, None]
    out = np.einsum("bk,b->k", big_g, dv * w)
    if s.periodic_u and face.window is None:
        # Loops that wrap around the axis (a band without seam edges, a cap): the
        # parameter domain, cut open along any u = const, gains two seam sides whose
        # integrals differ by a whole turn, H(v_end) - H(v_start) whatever the path;
        # summed over the wrapping loops (k turns each, at height v) that is
        # -sum k H(v), with a pole as a loop for a single cap.
        wraps = []
        for loop, sl in zip(face.loops, loop_slices, strict=True):
            uu = np.unwrap(uv[sl, 0])
            k = round(float(uu[-1] - uu[0]) / (2 * np.pi))
            if k:
                # The loop opens at its first vertex (between its last and first nodes).
                ei, fwd = loop[0]
                e = body.edges[ei]
                corner = body.vertices[e.v0 if fwd else e.v1]
                v = float(np.asarray(s.inverse(corner)).reshape(-1)[1])
                if s.periodic_v:
                    v0 = float(uv[sl.start, 1])
                    v = v0 + float(np.mod(v - v0 + np.pi, 2 * np.pi) - np.pi)
                wraps.append((k, v))
        if wraps:
            total_k = sum(k for k, _ in wraps)
            if total_k:
                above = (total_k > 0) == face.sense
                wraps.append((-total_k, _pole_v(s, above)))
            v_ref = wraps[0][1]
            for k, v in wraps[1:]:
                out -= k * _turn_integral(s, v_ref, v, center)
    # The area term must not depend on the loops' orientation: the face's area is
    # positive; with sense False the loops run clockwise in (u, v).
    out[10] = out[10] if face.sense else -out[10]
    return out


def mass_properties(body: Body, pieces: int = 2) -> MassProperties:
    if not body.faces:
        raise CadError("EMPTY_BODY", "the body has no faces")
    lo, hi = body.bounds
    center = 0.5 * (lo + hi)

    def total(k: int) -> np.ndarray:
        acc = np.zeros(11)
        for f in body.faces:
            acc += _face_integrals(body, f, center, k)
        return acc

    t1 = total(pieces)
    t2 = total(2 * pieces)
    err = np.abs(t2 - t1)
    vol, area = float(t2[0]), float(abs(t2[10]))
    if body.kind != "solid":
        vol = 0.0
    if body.kind == "solid" and vol <= 0:
        raise CadError(
            "INVALID_SOLID",
            f"the body encloses a non-positive volume ({vol:g}); its faces are not "
            "oriented outward",
        )
    if vol > 0:
        c_local = t2[1:4] / vol
        sxx, syy, szz, sxy, sxz, syz = t2[4], t2[5], t2[6], t2[7], t2[8], t2[9]
        # Second moments about the centroid (parallel axis theorem).
        cx, cy, cz = c_local
        sxx -= vol * cx * cx
        syy -= vol * cy * cy
        szz -= vol * cz * cz
        sxy -= vol * cx * cy
        sxz -= vol * cx * cz
        syz -= vol * cy * cz
        inertia = np.array(
            [
                [syy + szz, -sxy, -sxz],
                [-sxy, sxx + szz, -syz],
                [-sxz, -syz, sxx + syy],
            ]
        )
        evals, evecs = np.linalg.eigh(inertia)
        centroid = c_local + center
    else:
        inertia = np.zeros((3, 3))
        evals, evecs = np.zeros(3), np.eye(3)
        centroid = center.copy()
    return MassProperties(
        volume=vol,
        area=area,
        centroid=centroid,
        inertia=inertia,
        principal_moments=evals,
        principal_axes=evecs.T,
        error_volume=float(err[0]),
        error_area=float(err[10]),
    )


def face_area(body: Body, index: int) -> float:
    lo, hi = body.bounds
    out = _face_integrals(body, body.faces[index], 0.5 * (lo + hi), 2)
    return float(abs(out[10]))


def is_planar_body(body: Body) -> bool:
    return all(isinstance(f.surface, Plane) for f in body.faces)
