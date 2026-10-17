"""Boundary representations: bodies, shells, faces, loops, edges and vertices.

A :class:`Body` is an immutable B-rep:

* ``vertices`` -- (n, 3) float64 positions, with per-vertex tolerances;
* ``edges`` -- each an analytic or B-spline curve with a parameter range ``t0 < t1``, its
  start and end vertices (``curve.point(t0)``, ``curve.point(t1)``), a tolerance (0 for
  exact edges; the measured deviation for traced intersection curves) and a name;
* ``faces`` -- each a surface with a ``sense`` (``True``: the face's outward normal is
  the surface's ``Su x Sv``), loops of *coedges* ``(edge, forward)`` -- the outer loop
  first, counter-clockwise seen from outside, then inner loops clockwise -- a persistent
  name and a material group;
* ``shells`` -- face indices per closed shell (the first is the outer one; the others
  bound voids).

``kind`` is ``"solid"`` (closed shells bounding a volume) or ``"sheet"`` (open surfaces).
A body is never a triangle mesh: tessellations are derived from it
(:mod:`aether3d.cad.brep.tessellate`) and never replace it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...errors import CadError
from ..geom3d import (
    Circle3,
    Curve3,
    Ellipse3,
    Line3,
    Nurbs3,
    Surface,
    curve_from_dict,
    surface_from_dict,
)

TAU = 2.0 * math.pi
Coedge = tuple[int, bool]


@dataclass(frozen=True, eq=False)
class Edge:
    curve: Curve3
    t0: float
    t1: float
    v0: int
    v1: int
    name: str = ""
    tol: float = 0.0

    def points(self, ts) -> np.ndarray:
        return self.curve.point(ts)

    def midpoint(self) -> np.ndarray:
        return self.curve.point(0.5 * (self.t0 + self.t1))

    @property
    def length(self) -> float:
        c = self.curve
        if isinstance(c, Line3):
            return self.t1 - self.t0
        if isinstance(c, Circle3):
            return c.radius * (self.t1 - self.t0)
        ts = np.linspace(self.t0, self.t1, 513)
        p = c.point(ts)
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


@dataclass(frozen=True, eq=False)
class Face:
    surface: Surface
    sense: bool
    loops: tuple[tuple[Coedge, ...], ...]
    name: str = ""
    group: int = 0
    window: float | None = None
    """For periodic surfaces: the middle of the face's u range. A loop through poles (a
    sphere's meridians) does not tell which half turn it bounds; parameters are mapped
    into ``(window - pi, window + pi]``."""

    def to_window(self, uv: np.ndarray) -> np.ndarray:
        """Surface parameters with periodic u (and v) unwrapped consistently."""
        uv = np.array(uv, dtype=np.float64).reshape(-1, 2)
        s = self.surface
        if s.periodic_u:
            if self.window is not None:
                uv[:, 0] = self.window + np.mod(uv[:, 0] - self.window + math.pi, TAU) - math.pi
            else:
                uv[:, 0] = uv[0, 0] + np.unwrap(uv[:, 0] - uv[0, 0])
        if s.periodic_v:
            uv[:, 1] = uv[0, 1] + np.unwrap(uv[:, 1] - uv[0, 1])
        return uv


@dataclass(frozen=True, eq=False)
class Body:
    vertices: np.ndarray
    edges: tuple[Edge, ...]
    faces: tuple[Face, ...]
    shells: tuple[tuple[int, ...], ...]
    kind: str = "solid"
    vertex_tol: np.ndarray | None = None
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        v = np.array(self.vertices, dtype=np.float64).reshape(-1, 3)
        v.setflags(write=False)
        object.__setattr__(self, "vertices", v)
        tol = (
            np.zeros(len(v))
            if self.vertex_tol is None
            else np.array(self.vertex_tol, dtype=np.float64)
        )
        tol.setflags(write=False)
        object.__setattr__(self, "vertex_tol", tol)

    # -- queries --------------------------------------------------------------------

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = [self.vertices] if len(self.vertices) else []
        for e in self.edges:
            pts.append(e.curve.point(np.linspace(e.t0, e.t1, 9)))
        allp = np.concatenate(pts) if pts else np.zeros((1, 3))
        return allp.min(axis=0), allp.max(axis=0)

    @property
    def size(self) -> float:
        lo, hi = self.bounds
        return float(np.linalg.norm(hi - lo))

    def face_index(self, name: str) -> list[int]:
        return [
            i for i, f in enumerate(self.faces) if f.name == name or f.name.split("#")[0] == name
        ]

    def edge_faces(self) -> dict[int, list[tuple[int, bool]]]:
        """Coedges of every edge: (face, forward)."""
        out: dict[int, list[tuple[int, bool]]] = {}
        for fi, f in enumerate(self.faces):
            for loop in f.loops:
                for e, fwd in loop:
                    out.setdefault(e, []).append((fi, fwd))
        return out

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for f in self.faces:
            kinds[f.surface.kind] = kinds.get(f.surface.kind, 0) + 1
        curves: dict[str, int] = {}
        for e in self.edges:
            curves[e.curve.kind] = curves.get(e.curve.kind, 0) + 1
        return {
            "kind": self.kind,
            "faces": len(self.faces),
            "edges": len(self.edges),
            "vertices": len(self.vertices),
            "shells": len(self.shells),
            "surfaces": kinds,
            "curves": curves,
            "max_edge_tolerance": max((e.tol for e in self.edges), default=0.0),
            "max_vertex_tolerance": float(self.vertex_tol.max()) if len(self.vertex_tol) else 0.0,
        }

    # -- transforms -----------------------------------------------------------------

    def transformed(self, matrix: Any) -> Body:
        """The body under a rigid motion or reflection (4x4; uniform scale allowed)."""
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4) or not np.all(np.isfinite(m)):
            raise CadError("INVALID_TRANSFORM", "a body transform must be a finite 4x4 matrix")
        lin = m[:3, :3]
        det = float(np.linalg.det(lin))
        k = abs(det) ** (1.0 / 3.0)
        if k == 0 or not np.allclose(lin.T @ lin, k * k * np.eye(3), atol=1e-12 * k * k):
            raise CadError(
                "INVALID_TRANSFORM", "only rigid motions, reflections and uniform scales"
            )
        verts = self.vertices @ lin.T + m[:3, 3]
        edges = []
        swapped: set[int] = set()
        for i, e in enumerate(self.edges):
            c = e.curve.transformed(m)
            p0, p1, pm = verts[e.v0], verts[e.v1], (lin @ e.midpoint() + m[:3, 3])
            t0, t1 = edge_range(c, p0, pm, p1, closed=(e.v0 == e.v1))
            v0, v1 = e.v0, e.v1
            q0 = c.point(t0)
            if e.v0 != e.v1 and np.linalg.norm(q0 - p1) < np.linalg.norm(q0 - p0):
                # The curve's parameter now runs the other way (a reflection).
                v0, v1 = v1, v0
                swapped.add(i)
            edges.append(Edge(c, t0, t1, v0, v1, e.name, e.tol * k))
        faces = []
        for f in self.faces:
            s = f.surface.transformed(m)
            # Sense: compare the transformed outward normal with the new surface normal.
            pt, n = face_sample(self, f)
            pt2 = lin @ pt + m[:3, 3]
            n2 = lin @ n
            n2 = n2 / np.linalg.norm(n2)
            # (A reflection maps outward normals to outward normals of the image.)
            uv = s.inverse(pt2[None])[0]
            ns = s.normal(uv[0], uv[1])
            sense = bool(float(np.dot(ns.reshape(3), n2)) > 0)
            loops = tuple(
                tuple((ei, (not fwd) if ei in swapped else fwd) for ei, fwd in lp) for lp in f.loops
            )
            if det < 0:
                loops = tuple(tuple((ei, not fwd) for ei, fwd in reversed(lp)) for lp in loops)
            window = None
            if s.periodic_u:
                window = float(s.inverse(pt2[None])[0, 0])
            faces.append(Face(s, sense, loops, f.name, f.group, window))
        return Body(
            verts,
            tuple(edges),
            tuple(faces),
            self.shells,
            self.kind,
            self.vertex_tol * k,
            self.notes,
        )

    def renamed(self, prefix_map: dict[str, str]) -> Body:
        def rename(name: str) -> str:
            head, _, rest = name.partition(":")
            return f"{prefix_map.get(head, head)}:{rest}" if rest else prefix_map.get(name, name)

        faces = tuple(
            Face(f.surface, f.sense, f.loops, rename(f.name), f.group) for f in self.faces
        )
        return Body(
            self.vertices, self.edges, faces, self.shells, self.kind, self.vertex_tol, self.notes
        )

    # -- serialization --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "vertices": self.vertices.tolist(),
            "vertex_tol": self.vertex_tol.tolist(),
            "edges": [
                {
                    "curve": e.curve.to_dict(),
                    "t": [e.t0, e.t1],
                    "v": [e.v0, e.v1],
                    "name": e.name,
                    "tol": e.tol,
                }
                for e in self.edges
            ],
            "faces": [
                {
                    "surface": f.surface.to_dict(),
                    "sense": f.sense,
                    "loops": [[[ei, fwd] for ei, fwd in lp] for lp in f.loops],
                    "name": f.name,
                    "group": f.group,
                    "window": f.window,
                }
                for f in self.faces
            ],
            "shells": [list(s) for s in self.shells],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Body:
        edges = tuple(
            Edge(
                curve_from_dict(e["curve"]),
                float(e["t"][0]),
                float(e["t"][1]),
                int(e["v"][0]),
                int(e["v"][1]),
                str(e.get("name", "")),
                float(e.get("tol", 0.0)),
            )
            for e in d["edges"]
        )
        faces = tuple(
            Face(
                surface_from_dict(f["surface"]),
                bool(f["sense"]),
                tuple(tuple((int(a), bool(b)) for a, b in lp) for lp in f["loops"]),
                str(f.get("name", "")),
                int(f.get("group", 0)),
                None if f.get("window") is None else float(f["window"]),
            )
            for f in d["faces"]
        )
        return cls(
            np.asarray(d["vertices"], dtype=np.float64).reshape(-1, 3),
            edges,
            faces,
            tuple(tuple(int(i) for i in s) for s in d["shells"]),
            d.get("kind", "solid"),
            np.asarray(d.get("vertex_tol", [0.0] * len(d["vertices"])), dtype=np.float64),
        )


def edge_range(curve: Curve3, p0, pm, p1, *, closed: bool = False) -> tuple[float, float]:
    """Parameters ``t0 < t1`` of an edge from its end points and a point between them."""
    if isinstance(curve, Line3):
        t0 = float(curve.inverse(np.asarray(p0))[()])
        t1 = float(curve.inverse(np.asarray(p1))[()])
        return (t0, t1) if t0 <= t1 else (t1, t0)
    if isinstance(curve, (Circle3, Ellipse3)):
        a0 = float(curve.inverse(np.asarray(p0))[()])
        am = float(curve.inverse(np.asarray(pm), near=a0)[()])
        a1 = float(curve.inverse(np.asarray(p1), near=am)[()])
        if closed:
            # Full circle: direction from the midpoint.
            return (a0, a0 + TAU) if am > a0 else (a0 - TAU, a0)
        if am > a0:
            while a1 <= am:
                a1 += TAU
            return a0, a1
        while a1 >= am:
            a1 -= TAU
        return a1, a0
    if isinstance(curve, Nurbs3):
        t0 = float(curve.inverse(np.asarray(p0))[0])
        t1 = float(curve.inverse(np.asarray(p1))[0])
        return (t0, t1) if t0 <= t1 else (t1, t0)
    raise CadError("INVALID_GEOMETRY", f"unknown curve {curve!r}")


def infer_window(body: Body, face: Face) -> float | None:
    """The u window of a face on a u-periodic surface, from its loops: the window (among
    candidates around the turn) in which the outer loop unwraps into a closed polygon
    oriented as the face's sense requires (counter-clockwise in (u, v) when the face's
    normal is the surface's). Needed for faces bounded through poles (sphere halves
    between two meridians), which a boundary alone does not place."""
    s = face.surface
    if not s.periodic_u or not face.loops:
        return None
    pts = np.concatenate([coedge_points(body, c, None) for c in face.loops[0]])
    uv = np.array(s.inverse(pts), dtype=np.float64).reshape(-1, 2)
    su, _ = s.derivatives(uv[:, 0], uv[:, 1])
    good = np.linalg.norm(su, axis=1) > 1e-9 * max(body.size, 1e-9)
    if np.any(good):
        cont = np.unwrap(uv[good, 0])
        if float(cont.max() - cont.min()) >= TAU - 1e-6:
            return None  # a full turn (seam edges): unwrapped continuously, no window
    best = None
    for w in np.linspace(-math.pi, math.pi, 24, endpoint=False):
        u = w + np.mod(uv[:, 0] - w + math.pi, TAU) - math.pi
        u = np.where(good, u, np.nan)
        # Pole points take the u of their neighbours (they lie on a degenerate edge).
        if np.any(~good):
            idx = np.flatnonzero(good)
            if not len(idx):
                return None
            u = np.interp(np.arange(len(u)), idx, u[idx])
        v = uv[:, 1]
        if s.periodic_v:
            v = v[0] + np.unwrap(v - v[0])
        # Without poles the loop must close in (u, v); through poles it closes along
        # their degenerate edges.
        if np.all(good) and abs(u[0] - u[-1]) > 1.0:
            continue
        area = 0.5 * float(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v))
        if (area > 0) == face.sense and abs(area) > 0:
            span = float(u.max() - u.min())
            if span < TAU - 1e-9 and (best is None or abs(area) < best[1]):
                best = (float(0.5 * (u.max() + u.min())), abs(area))
    return None if best is None else best[0]


def coedge_points(body: Body, coedge: Coedge, ts: np.ndarray | None = None) -> np.ndarray:
    e = body.edges[coedge[0]]
    if ts is None:
        ts = np.linspace(e.t0, e.t1, 9)
    pts = e.curve.point(ts)
    return pts if coedge[1] else pts[::-1]


def face_sample(body: Body, face: Face) -> tuple[np.ndarray, np.ndarray]:
    """A point inside the face's surface region (near its outer loop's centroid in
    parameter space) and the face's outward normal there."""
    s = face.surface
    pts = np.concatenate([coedge_points(body, c) for c in face.loops[0]])
    uv = face.to_window(s.inverse(pts))
    center = uv.mean(axis=0)
    p = s.point(center[0], center[1]).reshape(3)
    n = s.normal(center[0], center[1]).reshape(3)
    return p, (n if face.sense else -n)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(body: Body, tol: float | None = None) -> dict[str, Any]:
    """Topological and geometric checks; returns a report (``valid`` and ``problems``).

    * every loop is closed (each coedge ends where the next starts);
    * in a solid, every edge has exactly two coedges, in opposite directions;
    * edge end points lie on their curves, curve samples on both faces' surfaces;
    * the Euler-Poincare relation holds per shell (V - E + F - (L - F) = 2 (S - G)).
    """
    problems: list[dict] = []
    size = max(body.size, 1e-3)
    if tol is None:
        tol = max(1e-9 * size, 1e-12)
    for i, e in enumerate(body.edges):
        p0, p1 = e.curve.point(e.t0), e.curve.point(e.t1)
        d0 = float(np.linalg.norm(p0 - body.vertices[e.v0]))
        d1 = float(np.linalg.norm(p1 - body.vertices[e.v1]))
        lim = tol + e.tol + float(max(body.vertex_tol[e.v0], body.vertex_tol[e.v1]))
        if max(d0, d1) > lim:
            problems.append(
                {"code": "VERTEX_OFF_CURVE", "edge": i, "distance": max(d0, d1), "name": e.name}
            )
    for fi, f in enumerate(body.faces):
        for li, loop in enumerate(f.loops):
            if not loop:
                problems.append({"code": "EMPTY_LOOP", "face": fi})
                continue
            for k, (ei, fwd) in enumerate(loop):
                e = body.edges[ei]
                end = e.v1 if fwd else e.v0
                nxt_e, nxt_fwd = loop[(k + 1) % len(loop)]
                ne = body.edges[nxt_e]
                start = ne.v0 if nxt_fwd else ne.v1
                if (
                    end != start
                    and float(np.linalg.norm(body.vertices[end] - body.vertices[start])) > tol
                ):
                    problems.append(
                        {"code": "LOOP_NOT_CLOSED", "face": fi, "loop": li, "at": k, "name": f.name}
                    )
            # Edges on the surface.
            for ei, _ in loop:
                e = body.edges[ei]
                pts = e.curve.point(np.linspace(e.t0, e.t1, 5))
                dist = float(np.max(f.surface.distance(pts)))
                if dist > tol + e.tol:
                    problems.append(
                        {
                            "code": "EDGE_OFF_SURFACE",
                            "face": fi,
                            "edge": ei,
                            "distance": dist,
                            "name": f.name,
                        }
                    )
    uses = body.edge_faces()
    if body.kind == "solid":
        for ei in range(len(body.edges)):
            u = uses.get(ei, [])
            if len(u) != 2:
                problems.append({"code": "NON_MANIFOLD_EDGE", "edge": ei, "uses": len(u)})
            elif u[0][1] == u[1][1]:
                problems.append({"code": "INCONSISTENT_ORIENTATION", "edge": ei})
    # Euler-Poincare per body.
    v = len({x for e in body.edges for x in (e.v0, e.v1)})
    e = len(body.edges)
    f = len(body.faces)
    loops = sum(len(fc.loops) for fc in body.faces)
    s = len(body.shells)
    chi = v - e + f - (loops - f)
    genus2 = 2 * s - chi
    euler_ok = genus2 >= 0 and genus2 % 2 == 0
    if body.kind == "solid" and not euler_ok:
        problems.append({"code": "EULER_VIOLATION", "chi": chi, "shells": s})
    return {
        "valid": not problems,
        "problems": problems[:50],
        "problem_count": len(problems),
        "euler": {
            "V": v,
            "E": e,
            "F": f,
            "L": loops,
            "S": s,
            "genus": genus2 // 2 if euler_ok else None,
        },
        "tolerance": tol,
    }
