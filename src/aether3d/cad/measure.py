"""Measurements of exact geometry, and snapping.

**References** (lengths in the document's unit)::

    {"part": "Bracket", "face": "base:end"}               # body "main" unless "body"
    {"part": "Bracket", "edge": ["base:end", "base:side/r2"]}
    {"part": "Bracket", "vertex": [x, y, z]}              # the vertex nearest the point
    {"sketch": "Profile", "entity": "l1"}                 # a line, arc or circle
    {"sketch": "Profile", "point": "p3"}
    {"point": [x, y, z]}

``doc.measure(a)`` describes one entity exactly: a face's surface type, area and (for
planes) normal, (for cylinders, cones, spheres, tori) radii and axis; an edge's curve
type, length and (circles) radius and centre; a vertex's position.

``doc.measure(a, b)`` relates two entities:

* ``angle`` -- between plane normals and/or line directions, exact;
* ``parallel_distance`` -- between parallel planes, parallel lines, or a point and a plane
  or line, exact (the distance between the infinite carriers);
* ``axis_distance`` -- between the axes of circles, cylinders, cones and spheres (centres),
  exact;
* ``distance`` -- the minimum distance between the two (trimmed) entities: the closest
  pair of samples of their exact geometry (edges sampled along their curves, faces at
  their tessellation's vertices and triangle centres within ``ANALYSIS`` tolerance),
  then refined by alternating projections onto the exact curves and surfaces. Reported
  with ``"method": "sampled+refined"`` and ``"bound"`` -- the sampling chord, an upper
  bound of how far the refined pair can be from the true closest pair when the closest
  points lie inside faces (it is exact when both closest points lie on edges or
  vertices refined onto their curves). A refined point that leaves a trimmed face is
  moved to the face's boundary.

**Snapping** (``doc.snap(sketch, x, y, radius=..., kinds=...)``): the sketch candidate
nearest the cursor within ``radius`` of the highest-priority kind present: end points,
then intersections, centres, midpoints, quadrants, the nearest point on a curve, and
last the grid (``grid`` spacing; the nearest grid point, whatever the radius). Results
are exact (closed-form intersections of lines and circles).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import CadError
from .brep.properties import face_area
from .brep.tessellate import tessellate
from .brep.topology import Body
from .geom2d import intersect
from .geom3d import Circle3, Line3, Plane
from .naming import resolve_edge, resolve_face
from .sketch.profile import sketch_curves
from .tolerance import ANALYSIS
from .units import LENGTH

SNAP_KINDS = ("endpoint", "intersection", "center", "midpoint", "quadrant", "nearest", "grid")
MAX_SNAP_CURVES = 5000


class _Entity:
    def __init__(self, kind: str, label: str, **data: Any) -> None:
        self.kind = kind  # face | edge | vertex | point | curve2d
        self.label = label
        self.__dict__.update(data)


def _unit(doc) -> float:
    return float(doc.evaluate_value(1.0, LENGTH, "unit"))


def _point(doc, raw, loc) -> np.ndarray:
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        raise CadError("INVALID_VALUE", "a point is [x, y, z]", loc)
    p = [doc.evaluate_value(v, LENGTH, f"{loc}[{i}]") for i, v in enumerate(raw)]
    return np.array(p + [0.0] * (3 - len(p)))


def resolve(doc, ref: Any, loc: str = "reference") -> _Entity:
    if not isinstance(ref, dict):
        raise CadError(
            "INVALID_REFERENCE",
            "a reference is an object",
            loc,
            hint='{"part": ..., "face": ...}, {"sketch": ..., "entity": ...}, {"point": [x, y, z]}',
        )
    s = _unit(doc)
    if "part" in ref:
        item = doc.item(ref["part"], "part")
        body_name = ref.get("body", "main")
        body = doc.result(item).body(body_name)
        label = f"{item.name}/{body_name}"
        if "face" in ref:
            fref = ref["face"]
            if "near" in ref and isinstance(fref, str):
                fref = {"face": fref, "near": ref["near"]}
            fi = resolve_face(body, fref, f"{loc}.face", s)
            return _Entity("face", f"{label}:{body.faces[fi].name}", body=body, index=fi)
        if "edge" in ref:
            eref = ref["edge"]
            if isinstance(eref, list):
                eref = {"edge": eref, **({"near": ref["near"]} if "near" in ref else {})}
            ei = resolve_edge(body, eref, f"{loc}.edge", s)
            return _Entity("edge", f"{label}:edge", body=body, index=ei)
        if "vertex" in ref:
            p = _point(doc, ref["vertex"], f"{loc}.vertex")
            d = np.linalg.norm(body.vertices - p, axis=1)
            vi = int(np.argmin(d))
            return _Entity("vertex", f"{label}:vertex", position=body.vertices[vi].copy())
        raise CadError("INVALID_REFERENCE", "a part reference names a face, edge or vertex", loc)
    if "sketch" in ref:
        res = doc.result(doc.item(ref["sketch"], "sketch"))
        if "point" in ref:
            return _Entity(
                "vertex", f"{ref['sketch']}:{ref['point']}", position=res.point3(ref["point"])
            )
        if "entity" in ref:
            curves = dict(sketch_curves(res.definition))
            c = curves.get(ref["entity"])
            if c is None:
                raise CadError(
                    "UNKNOWN_ENTITY",
                    f"the sketch has no curve {ref['entity']!r}",
                    loc,
                    hint=", ".join(curves),
                )
            return _Entity("curve2d", f"{ref['sketch']}:{ref['entity']}", curve=c, frame=res.frame)
        raise CadError("INVALID_REFERENCE", "a sketch reference names a point or an entity", loc)
    if "point" in ref:
        return _Entity("vertex", "point", position=_point(doc, ref["point"], f"{loc}.point"))
    raise CadError("INVALID_REFERENCE", "unknown reference", loc)


def _carrier(e: _Entity):
    """The exact carrier: ("plane", point, normal) | ("line", point, dir) |
    ("axis", point, dir) for axial things | ("point", p) or None."""
    if e.kind == "vertex":
        return ("point", e.position)
    if e.kind == "face":
        f = e.body.faces[e.index]
        srf = f.surface
        if isinstance(srf, Plane):
            n = srf.frame.z if f.sense else -srf.frame.z
            return ("plane", srf.frame.origin, n)
        if srf.kind in ("cylinder", "cone", "torus", "revolution", "extrusion"):
            return ("axis", srf.frame.origin, srf.frame.z)
        if srf.kind == "sphere":
            return ("center", srf.frame.origin)
        return None
    if e.kind == "edge":
        c = e.body.edges[e.index].curve
        if isinstance(c, Line3):
            return ("line", c.origin, c.direction / np.linalg.norm(c.direction))
        if isinstance(c, Circle3):
            return ("circle", c.frame.origin, c.frame.z)
        return None
    if e.kind == "curve2d":
        c = e.curve
        fr = e.frame
        if c.kind == "segment":
            a = fr.to_world(np.array([*c.p0, 0.0]))
            b = fr.to_world(np.array([*c.p1, 0.0]))
            return ("line", a, (b - a) / np.linalg.norm(b - a))
        if c.kind == "arc":
            return ("circle", fr.to_world(np.array([*c.center, 0.0])), fr.z)
    return None


def describe(doc, e: _Entity) -> dict[str, Any]:
    s = _unit(doc)
    out: dict[str, Any] = {"reference": e.label, "kind": e.kind}
    if e.kind == "vertex":
        out["position"] = (e.position / s).tolist()
    elif e.kind == "face":
        f = e.body.faces[e.index]
        srf = f.surface
        out["surface"] = srf.kind
        out["area"] = face_area(e.body, e.index) / (s * s)
        if isinstance(srf, Plane):
            out["normal"] = (srf.frame.z if f.sense else -srf.frame.z).tolist()
        for attr in ("radius", "major", "minor"):
            if hasattr(srf, attr):
                out[attr] = float(getattr(srf, attr)) / s
        if srf.kind in ("cylinder", "cone", "sphere", "torus", "revolution"):
            out["axis"] = {
                "point": (srf.frame.origin / s).tolist(),
                "direction": srf.frame.z.tolist(),
            }
        if srf.kind == "cone":
            out["half_angle_deg"] = math.degrees(srf.half_angle)
    elif e.kind == "edge":
        edge = e.body.edges[e.index]
        out["curve"] = edge.curve.kind
        out["length"] = edge.length / s
        if isinstance(edge.curve, Circle3):
            out["radius"] = edge.curve.radius / s
            out["center"] = (edge.curve.frame.origin / s).tolist()
    elif e.kind == "curve2d":
        c = e.curve
        out["curve"] = c.kind
        out["length"] = c.length / s
        if c.kind == "arc":
            out["radius"] = c.radius / s
            out["sweep_deg"] = math.degrees(c.sweep)
    return out


def _samples(e: _Entity) -> np.ndarray:
    if e.kind == "vertex":
        return e.position[None]
    if e.kind == "face":
        body = e.body
        sub = Body(
            body.vertices, body.edges, (body.faces[e.index],), ((0,),), "sheet", body.vertex_tol
        )
        t = tessellate(sub, ANALYSIS)
        tri = t.positions[t.triangles]
        return np.concatenate([t.positions[np.unique(t.triangles)], tri.mean(axis=1)])
    if e.kind == "edge":
        edge = e.body.edges[e.index]
        return edge.curve.point(np.linspace(edge.t0, edge.t1, 513))
    c = e.curve
    pts = c.point(np.linspace(0.0, 1.0, 513))
    return np.array([e.frame.to_world(np.array([p[0], p[1], 0.0])) for p in pts])


def _project(e: _Entity, p: np.ndarray) -> np.ndarray:
    """The closest point of ``e``'s exact geometry to ``p`` (locally)."""
    if e.kind == "vertex":
        return e.position
    if e.kind == "face":
        face = e.body.faces[e.index]
        srf = face.surface
        uv = srf.inverse(p[None])
        q = srf.point(uv[:, 0], uv[:, 1]).reshape(3)
        if _inside_face(e.body, face, uv[0]):
            return q
        # Outside the trimmed face: the closest point is on its boundary.
        best = None
        for loop in face.loops:
            for ei, _ in loop:
                c = _project(_Entity("edge", "", body=e.body, index=ei), p)
                d = float(np.linalg.norm(c - p))
                if best is None or d < best[0]:
                    best = (d, c)
        return best[1]
    if e.kind == "edge":
        edge = e.body.edges[e.index]
        c = edge.curve
        ts = np.linspace(edge.t0, edge.t1, 65)
        k = int(np.argmin(np.linalg.norm(c.point(ts) - p, axis=1)))
        lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
        for _ in range(60):  # golden-section on the bracket
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            d1 = np.linalg.norm(c.point(np.array([m1]))[0] - p)
            d2 = np.linalg.norm(c.point(np.array([m2]))[0] - p)
            if d1 < d2:
                hi = m2
            else:
                lo = m1
        return c.point(np.array([0.5 * (lo + hi)]))[0]
    c = e.curve
    q = e.frame.to_local(p)
    _, cp = c.closest(q[:2])
    return e.frame.to_world(np.array([cp[0], cp[1], 0.0]))


def _inside_face(body: Body, face, uv: np.ndarray) -> bool:
    """Whether surface parameters ``uv`` lie inside the face's loops (even-odd)."""
    from .brep.topology import coedge_points

    srf = face.surface
    loops = []
    for loop in face.loops:
        pts = np.concatenate([coedge_points(body, c, None) for c in loop])
        loops.append(np.array(srf.inverse(pts), dtype=np.float64).reshape(-1, 2))
    allp = face.to_window(np.concatenate(loops))
    k = 0
    for i, lp in enumerate(loops):
        loops[i] = allp[k : k + len(lp)]
        k += len(lp)
    u, v = float(uv[0]), float(uv[1])
    if srf.periodic_u:
        lo, hi = float(allp[:, 0].min()), float(allp[:, 0].max())
        u = lo + (u - lo) % (2 * math.pi) if not lo <= u <= hi else u
    inside = False
    for lp in loops:
        x, y = lp[:, 0], lp[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        cross = (y > v) != (y2 > v)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x + (v - y) * (x2 - x) / (y2 - y)
        inside ^= bool(np.count_nonzero(cross & (u < xc)) % 2)
    return inside


def relate(doc, a: _Entity, b: _Entity) -> dict[str, Any]:
    s = _unit(doc)
    out: dict[str, Any] = {"a": a.label, "b": b.label}
    ca, cb = _carrier(a), _carrier(b)
    dirs = {}
    for name, c in (("a", ca), ("b", cb)):
        if c is not None and c[0] in ("plane", "line"):
            dirs[name] = (c[0], c[2])
    if len(dirs) == 2:
        (ka, da), (kb, db) = dirs["a"], dirs["b"]
        cosv = abs(float(da @ db))
        if ka == kb:  # plane-plane (normals) or line-line (directions)
            ang = math.degrees(math.acos(min(1.0, cosv)))
        else:  # a line against a plane: 90 degrees minus the angle to the normal
            ang = 90.0 - math.degrees(math.acos(min(1.0, cosv)))
        out["angle_deg"] = ang
        parallel = (cosv > 1 - 1e-12) if ka == kb else (cosv < 1e-12)
        if parallel:
            if ka == "plane":
                out["parallel_distance"] = abs(float((cb[1] - ca[1]) @ ca[2])) / s
            elif ka == "line" and kb == "line":
                d = cb[1] - ca[1]
                out["parallel_distance"] = float(np.linalg.norm(d - (d @ da) * da)) / s
    for p, plane in ((ca, cb), (cb, ca)):
        if p is not None and plane is not None and p[0] == "point" and plane[0] == "plane":
            out["parallel_distance"] = abs(float((p[1] - plane[1]) @ plane[2])) / s
        if p is not None and plane is not None and p[0] == "point" and plane[0] == "line":
            d = p[1] - plane[1]
            out["parallel_distance"] = float(np.linalg.norm(d - (d @ plane[2]) * plane[2])) / s
    axial = ("axis", "circle", "center")
    if ca is not None and cb is not None and ca[0] in axial and cb[0] in axial:
        if ca[0] == "center" and cb[0] == "center":
            out["axis_distance"] = float(np.linalg.norm(cb[1] - ca[1])) / s
        elif ca[0] == "center" or cb[0] == "center":
            pt, ax = (ca, cb) if ca[0] == "center" else (cb, ca)
            d = pt[1] - ax[1]
            out["axis_distance"] = float(np.linalg.norm(d - (d @ ax[2]) * ax[2])) / s
        else:
            da, db = ca[2], cb[2]
            d = cb[1] - ca[1]
            n = np.cross(da, db)
            if np.linalg.norm(n) < 1e-12:
                out["axis_distance"] = float(np.linalg.norm(d - (d @ da) * da)) / s
            else:
                out["axis_distance"] = abs(float(d @ n)) / float(np.linalg.norm(n)) / s
            out["axis_angle_deg"] = math.degrees(math.acos(min(1.0, abs(float(da @ db)))))
    # Minimum distance between the trimmed entities.
    pa, pb = _samples(a), _samples(b)
    best = (math.inf, None, None)
    for start in range(0, len(pa), 2048):
        chunk = pa[start : start + 2048]
        d = np.linalg.norm(chunk[:, None, :] - pb[None, :, :], axis=2)
        k = int(np.argmin(d))
        i, j = divmod(k, d.shape[1])
        if d[i, j] < best[0]:
            best = (float(d[i, j]), chunk[i], pb[j])
    _, qa, qb = best
    for _ in range(50):
        na = _project(a, qb)
        nb = _project(b, na)
        if np.linalg.norm(na - qa) + np.linalg.norm(nb - qb) < 1e-15:
            qa, qb = na, nb
            break
        qa, qb = na, nb
    dist = float(np.linalg.norm(qb - qa))
    if dist > best[0]:  # refinement left the trimmed region: keep the sampled pair
        dist, qa, qb = best
    bound = 0.0
    for e in (a, b):
        if e.kind == "face":
            bound = max(bound, ANALYSIS.chord_for(e.body.size))
    out["distance"] = dist / s
    out["closest"] = [(qa / s).tolist(), (qb / s).tolist()]
    out["method"] = "sampled+refined"
    out["bound"] = bound / s
    return out


def measure(doc, a: Any, b: Any = None) -> dict[str, Any]:
    ea = resolve(doc, a, "a")
    if b is None:
        return describe(doc, ea)
    return relate(doc, ea, resolve(doc, b, "b"))


# --------------------------------------------------------------------------- #
# Snapping
# --------------------------------------------------------------------------- #


def snap(
    doc, sketch: Any, x: Any, y: Any, *, radius: Any = None, kinds=SNAP_KINDS, grid: Any = None
) -> dict[str, Any] | None:
    """The best snap candidate for the cursor (x, y) in a sketch's plane (document
    units), or None. ``radius`` defaults to 2 % of the sketch's extent."""
    res = doc.result(doc.item(sketch, "sketch"))
    s = _unit(doc)
    cur = np.array([doc.evaluate_value(x, LENGTH, "x"), doc.evaluate_value(y, LENGTH, "y")])
    for k in kinds:
        if k not in SNAP_KINDS:
            raise CadError("INVALID_VALUE", f"unknown snap kind {k!r}", hint=", ".join(SNAP_KINDS))
    curves = sketch_curves(res.definition)
    if len(curves) > MAX_SNAP_CURVES:
        raise CadError("LIMIT_EXCEEDED", f"snapping looks at most {MAX_SNAP_CURVES} curves")
    pts = [np.asarray(e["at"], float) for e in res.definition["entities"] if e["type"] == "point"]
    if radius is None:
        allp = np.array(pts) if pts else cur[None]
        radius_m = max(float(np.ptp(allp, axis=0).max()) * 0.02, 1e-9)
    else:
        radius_m = doc.evaluate_value(radius, LENGTH, "radius")
    cands: dict[str, list[tuple[np.ndarray, str]]] = {k: [] for k in SNAP_KINDS}
    for cid, c in curves:
        cands["endpoint"] += (
            [(c.start, cid), (c.end, cid)]
            if getattr(c, "kind", "") != "arc" or not c.closed
            else []
        )
        if c.kind == "segment":
            cands["midpoint"].append((0.5 * (c.p0 + c.p1), cid))
        elif c.kind == "arc":
            cands["center"].append((c.center, cid))
            if not c.closed:
                cands["midpoint"].append((c.mid, cid))
            for q in range(4):
                ang = q * math.pi / 2
                t = c.param_of_angle(ang) if not c.closed else ang / (2 * math.pi)
                if c.closed or 0.0 <= t <= 1.0:
                    cands["quadrant"].append(
                        (c.center + c.radius * np.array([math.cos(ang), math.sin(ang)]), cid)
                    )
        if hasattr(c, "closest"):
            _, near = c.closest(cur)
            cands["nearest"].append((near, cid))
    if "intersection" in kinds:
        tol = 1e-9 * max(radius_m * 50, 1e-9)
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                a, b = curves[i][1], curves[j][1]
                if a.kind == "spline" or b.kind == "spline":
                    continue
                for h in intersect(a, b, tol):
                    cands["intersection"].append((h.point, f"{curves[i][0]}x{curves[j][0]}"))
    if grid is not None:
        g = doc.evaluate_value(grid, LENGTH, "grid")
        if g <= 0:
            raise CadError("INVALID_VALUE", "grid spacing must be positive")
        cands["grid"].append((np.round(cur / g) * g, "grid"))
    for k in kinds:
        best = None
        for p, ref in cands[k]:
            d = float(np.linalg.norm(p - cur))
            if (d <= radius_m or k == "grid") and (best is None or d < best[0]):
                best = (d, p, ref)
        if best is not None:
            return {
                "kind": k,
                "point": (best[1] / s).tolist(),
                "ref": best[2],
                "distance": best[0] / s,
            }
    return None


def _install() -> None:
    from .document import CadDocument

    CadDocument.measure = measure
    CadDocument.snap = snap


_install()
