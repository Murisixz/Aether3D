"""Tessellation of B-reps: watertight triangle meshes within explicit tolerances.

Tessellations are *derived* data with a stated accuracy, never the model:

1. **Edges first.** Every edge is sampled once -- lines at their ends, circles and
   ellipses by the chord/angle formulas, B-splines by bisection -- and both faces of an
   edge use exactly those samples, so the mesh is watertight (no cracks, no T-junctions)
   and its boundary vertices are the B-rep's vertices.
2. **Planar faces** are triangulated in their plane (ear clipping with holes).
3. **Curved faces** are triangulated in their parameter domain: the boundary samples
   mapped to (u, v) (periodic parameters unwrapped along the loops, poles of spheres and
   cones split into their two directions), ear clipping, Delaunay flips in a metric that
   follows the surface's scale, then refinement -- a point inserted inside every triangle
   whose flat approximation deviates from the surface by more than the chordal tolerance
   or whose corner normals differ by more than the angle tolerance -- until none does.
4. Normals are the exact surface normals; texture coordinates are surface parameters
   scaled to metres; each triangle remembers its face.

``chord`` reported by the result is the deviation the tessellation was built to; the
measured deviation (``max_deviation``) is computed on request by sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...errors import CadError
from ..geom3d import Circle3, Ellipse3, Line3, Nurbs3, Plane
from ..mesh2d import TriMesh2D
from ..tolerance import DISPLAY, TessellationTolerance
from ..triangulate import triangulate
from .topology import Body

MAX_TRIANGLES = 4_000_000
MAX_FACE_POINTS = 200_000


@dataclass
class Tessellation:
    positions: np.ndarray  # (V, 3) float64, shared between faces
    triangles: np.ndarray  # (T, 3) int64, counter-clockwise seen from outside
    face_of: np.ndarray  # (T,) int64
    corner_normals: np.ndarray  # (T, 3, 3)
    corner_uv: np.ndarray  # (T, 3, 2) surface parameters in metres
    chord: float
    angle: float
    edge_samples: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    def to_mesh(self, groups: np.ndarray | None = None):
        """An Aether3D render :class:`~aether3d.Mesh` (vertices split per face)."""
        from ...mesh import Mesh

        t = len(self.triangles)
        if t == 0:
            raise CadError("EMPTY_TESSELLATION", "nothing to tessellate")
        # One vertex per (face, position) so normals stay smooth inside faces and sharp
        # between them.
        face_rep = np.repeat(self.face_of, 3)
        pos_id = self.triangles.reshape(-1)
        key = face_rep.astype(np.int64) * (len(self.positions) + 1) + pos_id
        uniq, inverse = np.unique(key, return_inverse=True)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(key))[::-1]
        positions = self.positions[pos_id[first]].astype(np.float32)
        normals = self.corner_normals.reshape(-1, 3)[first].astype(np.float32)
        uvs = self.corner_uv.reshape(-1, 2)[first].astype(np.float32)
        tris = inverse.reshape(-1, 3).astype(np.uint32)
        g = None
        if groups is not None:
            g = np.asarray(groups, dtype=np.uint16)[self.face_of]
        return Mesh(positions, tris, normals=normals, uvs=uvs, groups=g)


def _edge_params(
    edge, chord: float, angle: float, max_edge: float | None, wide: float = 0.0
) -> np.ndarray:
    """Parameters sampling ``edge``; ``wide`` is the largest radius of a cone or cylinder
    band the edge bounds (its angular steps must also suit that band's wide end)."""
    c = edge.curve
    span = edge.t1 - edge.t0
    if isinstance(c, Line3):
        n = 1
        if max_edge:
            n = max(1, math.ceil(span / max_edge))
        return np.linspace(edge.t0, edge.t1, n + 1)
    if isinstance(c, (Circle3, Ellipse3)):
        r = c.radius if isinstance(c, Circle3) else c.minor * c.minor / c.major
        r_max = c.radius if isinstance(c, Circle3) else c.major
        if isinstance(c, Circle3) and wide > r:
            r = wide
        step = 2.0 * math.acos(max(-1.0, 1.0 - chord / r)) if chord < r else math.pi
        n = max(math.ceil(span / step), math.ceil(span / angle), 1)
        if max_edge:
            n = max(n, math.ceil(span * r_max / max_edge))
        return np.linspace(edge.t0, edge.t1, min(n, 1 << 16) + 1)
    if isinstance(c, Nurbs3):
        ts = list(
            np.linspace(edge.t0, edge.t1, 4 * max(1, len(c.curve.points) - c.curve.degree) + 1)
        )
        out = [ts[0]]
        stack = list(reversed(list(zip(ts[:-1], ts[1:], strict=True))))
        while stack and len(out) < 1 << 16:
            a, b = stack.pop()
            m = 0.5 * (a + b)
            pa, pm, pb = c.point(np.array([a, m, b]))
            dev = float(np.linalg.norm(pm - 0.5 * (pa + pb)))
            da, db = c.derivative(np.array([a, b]), 1)
            cosang = float(da @ db) / max(float(np.linalg.norm(da) * np.linalg.norm(db)), 1e-300)
            too_long = max_edge is not None and float(np.linalg.norm(pb - pa)) > max_edge
            if (dev > chord or cosang < math.cos(angle) or too_long) and b - a > 1e-12 * (
                edge.t1 - edge.t0
            ):
                stack += [(m, b), (a, m)]
            else:
                out.append(b)
        return np.array(out)
    raise CadError("INVALID_GEOMETRY", f"cannot sample {c!r}")


def tessellate(body: Body, tolerance: TessellationTolerance = DISPLAY) -> Tessellation:
    size = max(body.size, 1e-6)
    chord = tolerance.chord_for(size)
    angle = float(tolerance.angle)
    max_edge = tolerance.max_edge
    positions: list[np.ndarray] = [body.vertices.copy()]
    count = len(body.vertices)
    samples: dict[int, np.ndarray] = {}
    wide = _band_radii(body)
    fine = _doubly_curved_edges(body)
    for ei, e in enumerate(body.edges):
        c_edge = chord * 0.5 if ei in fine else chord
        ts = _edge_params(e, c_edge, angle, max_edge, wide.get(ei, 0.0))
        inner = e.curve.point(ts[1:-1]) if len(ts) > 2 else np.zeros((0, 3))
        ids = np.concatenate([[e.v0], np.arange(count, count + len(inner)), [e.v1]]).astype(
            np.int64
        )
        positions.append(inner)
        count += len(inner)
        samples[ei] = ids
    pos = np.concatenate(positions, axis=0)
    tris_all, faces_all, normals_all, uv_all = [], [], [], []
    extra_pos: list[np.ndarray] = []
    for fi, face in enumerate(body.faces):
        loops = []
        for loop in face.loops:
            ids = []
            for ei, fwd in loop:
                s = samples[ei]
                seq = s if fwd else s[::-1]
                ids.extend(seq[:-1].tolist())
            loops.append(np.array(ids, dtype=np.int64))
        if isinstance(face.surface, Plane):
            t, n, uv = _planar(face, loops, pos)
        else:
            t, n, uv, new_pts = _curved(face, loops, pos, count, chord, angle, size)
            if len(new_pts):
                extra_pos.append(new_pts)
                count += len(new_pts)
                pos = np.concatenate([pos, new_pts], axis=0)
        tris_all.append(t)
        faces_all.append(np.full(len(t), fi, dtype=np.int64))
        normals_all.append(n)
        uv_all.append(uv)
        if sum(len(x) for x in tris_all) > MAX_TRIANGLES:
            raise CadError(
                "TESSELLATION_TOO_LARGE",
                f"the tessellation exceeds {MAX_TRIANGLES} triangles",
                hint="use a coarser tolerance",
            )
    tris = np.concatenate(tris_all) if tris_all else np.zeros((0, 3), np.int64)
    return Tessellation(
        positions=pos,
        triangles=tris,
        face_of=np.concatenate(faces_all) if faces_all else np.zeros(0, np.int64),
        corner_normals=np.concatenate(normals_all) if normals_all else np.zeros((0, 3, 3)),
        corner_uv=np.concatenate(uv_all) if uv_all else np.zeros((0, 3, 2)),
        chord=chord,
        angle=angle,
        edge_samples=samples,
    )


def _band_radii(body: Body) -> dict[int, float]:
    """For circle edges on cone faces: the cone face's largest radius (coaxial circles of
    one band then get the same angular steps, so strips between them stay within the
    chord where the band is widest)."""
    out: dict[int, float] = {}
    for f in body.faces:
        s = f.surface
        if s.kind != "cone":
            continue
        edges = {ei for lp in f.loops for ei, _ in lp}
        r_max = 0.0
        for ei in edges:
            e = body.edges[ei]
            pts = e.curve.point(np.array([e.t0, 0.5 * (e.t0 + e.t1), e.t1]))
            loc = s.frame.to_local(pts)
            r_max = max(r_max, float(np.max(np.hypot(loc[:, 0], loc[:, 1]))))
        for ei in edges:
            if isinstance(body.edges[ei].curve, Circle3):
                out[ei] = max(out.get(ei, 0.0), r_max)
    return out


def _rings(loops: list[np.ndarray]):
    starts = [0]
    for lp in loops:
        starts.append(starts[-1] + len(lp))
    return np.array(starts, dtype=np.int64)


def _planar(face, loops, pos):
    fr = face.surface.frame
    ids = np.concatenate(loops)
    pts2 = fr.to_local(pos[ids])[:, :2]
    tri, _ = triangulate(pts2, _rings(loops), np.array([0, len(loops)]))
    g = ids[tri]
    if not face.sense:
        g = g[:, ::-1]
    normal = fr.z if face.sense else -fr.z
    n = np.broadcast_to(normal, (len(g), 3, 3)).copy()
    uv = fr.to_local(pos[g.reshape(-1)])[:, :2].reshape(-1, 3, 2)
    return g, n, uv


def _unwrap(vals: np.ndarray, ref: float | None) -> np.ndarray:
    out = vals[0] + np.unwrap(vals - vals[0])
    if ref is not None:
        out += 2 * math.pi * np.round((ref - out[0]) / (2 * math.pi))
    return out


def _curved(face, loops, pos, base, chord, angle, size):
    s = face.surface
    uv_loops = []
    pid_loops = []
    through_pole: list[bool] = []
    ref_u = ref_v = None
    for lp in loops:
        p = pos[lp]
        uv = np.array(s.inverse(p), dtype=np.float64).reshape(-1, 2)
        su, sv = s.derivatives(uv[:, 0], uv[:, 1])
        singular = np.linalg.norm(su, axis=1) <= 1e-9 * size
        if s.periodic_u:
            good = ~singular
            if np.any(good):
                if face.window is not None:
                    uv[good] = face.to_window(uv[good])
                else:
                    vals = uv[good, 0]
                    uv[good, 0] = _unwrap(vals, ref_u)
                    ref_u = float(uv[good, 0][0]) if ref_u is None else ref_u
        if s.periodic_v:
            uv[:, 1] = _unwrap(uv[:, 1], ref_v)
            ref_v = float(uv[0, 1]) if ref_v is None else ref_v
        if s.periodic_u and face.window is None and np.any(singular) and np.any(~singular):
            # A loop through a pole that wraps around the axis opens at the pole (its u
            # jumps by a turn there).
            k = _turns(uv[~singular, 0])
            if k:
                first = int(np.flatnonzero(singular)[0])
                after = np.arange(len(uv)) > first
                uv[after & ~singular, 0] -= 2 * math.pi * k
        # Poles: one parameter point per neighbouring direction.
        out_uv, out_id = [], []
        n = len(lp)
        for k in range(n):
            if singular[k]:
                prev_u = uv[(k - 1) % n, 0]
                next_u = uv[(k + 1) % n, 0]
                out_uv += [[prev_u, uv[k, 1]], [next_u, uv[k, 1]]]
                out_id += [lp[k], lp[k]]
            else:
                out_uv.append(uv[k].tolist())
                out_id.append(int(lp[k]))
        uv_loops.append(np.array(out_uv))
        pid_loops.append(np.array(out_id, dtype=np.int64))
        through_pole.append(bool(np.any(singular)))
    seam_pts = np.zeros((0, 3))
    if s.periodic_u and face.window is None:
        uv_loops, pid_loops, seam_pts = _virtual_seams(
            face, uv_loops, pid_loops, through_pole, base, chord, angle, size
        )
        if len(seam_pts):
            pos = np.concatenate([pos, seam_pts], axis=0)
            base += len(seam_pts)
    all_uv = np.concatenate(uv_loops)
    all_id = np.concatenate(pid_loops)
    nb = len(all_uv)
    c = all_uv.mean(axis=0)
    su, sv = s.derivatives(np.array([c[0]]), np.array([c[1]]))
    sx = max(float(np.linalg.norm(su)), 1e-12)
    sy = max(float(np.linalg.norm(sv)), 1e-12)
    band = rows = None
    if s.kind in ("cylinder", "cone", "extrusion") and len(uv_loops) == 1:
        band = _ruled_band(all_uv)
    elif s.kind in ("sphere", "torus", "revolution") and len(uv_loops) == 1:
        rows = _rows(s, all_uv, chord, angle)
    if band is not None:
        bottom, top = band
        pts = all_uv
        tris = np.array(_zip(all_uv[:, 0], bottom, top), dtype=np.int64)
    elif rows is not None:
        pts, tris = rows
    else:
        tri, _ = triangulate(all_uv, _rings(uv_loops), np.array([0, len(uv_loops)]))
        mesh = TriMesh2D(all_uv, tri, sx, sy)
        mesh.make_delaunay()
        _seed(mesh, s, uv_loops, chord, angle)
        _refine(mesh, s, all_id, pos, nb, chord, angle)
        pts = mesh.points()
        tris = mesh.triangles()
    ids = np.concatenate([all_id, np.arange(base, base + len(pts) - nb)])
    new_pts = s.point(pts[nb:, 0], pts[nb:, 1]) if len(pts) > nb else np.zeros((0, 3))
    new_pts = np.concatenate([seam_pts, new_pts], axis=0)
    g = ids[tris]
    keep = (g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 2] != g[:, 0])
    g, tl = g[keep], tris[keep]
    uvt = pts[tl]
    n = s.normal(uvt[..., 0], uvt[..., 1])
    if not face.sense:
        g = g[:, ::-1]
        n = -n[:, ::-1]
        uvt = uvt[:, ::-1]
    uvm = uvt * np.array([sx, sy])
    return g, n, uvm, new_pts


def _turns(u: np.ndarray) -> int:
    """Net turns of a closed loop's unwrapped u (including its closing step)."""
    w = np.unwrap(np.concatenate([u, u[:1]]))
    return round(float(w[-1] - w[0]) / (2 * math.pi))


def _seam_params(s, p, q, chord: float, angle: float, size: float) -> np.ndarray:
    """Interior parameters t in (0, 1) along the straight (u, v) segment p -> q, fine
    enough for the chord and angle tolerances."""
    ts = [0.0, 1.0]
    out = [0.0]
    stack = [(0.0, 1.0)]
    cos_a = math.cos(angle)
    while stack:
        a, b = stack.pop()
        m = 0.5 * (a + b)
        pa, pb, pm = (s.point(*(p + t * (q - p))).reshape(3) for t in (a, b, m))
        na, nb = (s.normal(*(p + t * (q - p))).reshape(3) for t in (a, b))
        dev = float(np.linalg.norm(pm - 0.5 * (pa + pb)))
        if (dev > chord or float(na @ nb) < cos_a) and b - a > 1e-6 and len(ts) < 4096:
            ts.append(m)
            stack += [(m, b), (a, m)]
        else:
            out.append(b)
    del size
    return np.array(out[1:-1])


def _virtual_seams(face, uv_loops, pid_loops, through_pole, base, chord, angle, size):
    """Faces on u-periodic surfaces whose loops wrap around the axis (a band between two
    loops, or a cap around a pole) have no seam edge to cut their parameter domain open.
    Join the wrapping loops by a seam curve (an iso-parametric segment in (u, v),
    sampled once and used by both sides) into one simple ring."""
    s = face.surface
    # Loops through a pole are already open there.
    wraps = [i for i, lp in enumerate(uv_loops) if not through_pole[i] and _turns(lp[:, 0])]
    if not wraps:
        return uv_loops, pid_loops, np.zeros((0, 3))
    if len(wraps) > 2:
        raise CadError("UNSUPPORTED_GEOMETRY", "a face wrapping around its axis more than twice")
    a = wraps[0]
    ua, ida = uv_loops[a], pid_loops[a]
    ka = _turns(ua[:, 0])
    end_a = ua[0] + np.array([2 * math.pi * ka, 0.0])
    if len(wraps) == 2:
        b = wraps[1]
        ub, idb = uv_loops[b].copy(), pid_loops[b]
        kb = _turns(ub[:, 0])
        if kb != -ka:
            raise CadError("UNSUPPORTED_GEOMETRY", "a face whose boundaries wrap the same way")
        ub[:, 0] += 2 * math.pi * round(float(end_a[0] - ub[0, 0]) / (2 * math.pi))
        target = ub[0]
    else:
        from .properties import _pole_v

        above = (ka > 0) == face.sense
        target = np.array([end_a[0], _pole_v(s, above)])
    ts = _seam_params(s, end_a, target, chord, angle, size)
    seam_uv = end_a[None] + ts[:, None] * (target - end_a)[None]
    seam_ids = np.arange(base, base + len(seam_uv))
    seam3 = s.point(seam_uv[:, 0], seam_uv[:, 1]).reshape(-1, 3)
    shift = np.array([2 * math.pi * ka, 0.0])
    if len(wraps) == 2:
        ring = [ua, end_a[None], seam_uv, ub, (ub[0] - shift)[None], seam_uv[::-1] - shift]
        ids = [ida, ida[:1], seam_ids, idb, idb[:1], seam_ids[::-1]]
    else:
        pole_id = base + len(seam_uv)
        seam3 = np.concatenate([seam3, s.point(target[0], target[1]).reshape(1, 3)])
        ring = [
            ua,
            end_a[None],
            seam_uv,
            target[None],
            (target - shift)[None],
            seam_uv[::-1] - shift,
        ]
        ids = [ida, ida[:1], seam_ids, [pole_id], [pole_id], seam_ids[::-1]]
    merged_uv = np.concatenate(ring)
    merged_id = np.concatenate([np.asarray(x, dtype=np.int64) for x in ids])
    rest = [i for i in range(len(uv_loops)) if i not in wraps]
    return (
        [merged_uv] + [uv_loops[i] for i in rest],
        [merged_id] + [pid_loops[i] for i in rest],
        seam3,
    )


def _ruled_band(uv: np.ndarray) -> np.ndarray | None:
    """Strip triangulation of a face whose loop is two u-monotone chains joined by two
    rulings (straight in space): returns triangles (local indices) or ``None``."""
    n = len(uv)
    u = uv[:, 0]
    span = float(u.max() - u.min())
    if n < 4 or span <= 0:
        return None
    eps = 1e-9 * max(span, 1.0)
    at_min = u <= u.min() + eps
    at_max = u >= u.max() - eps
    if np.count_nonzero(at_min) > 2 or np.count_nonzero(at_max) > 2:
        return None

    def run(mask: np.ndarray) -> tuple[int, int] | None:
        idx = np.flatnonzero(mask)
        if len(idx) == 1:
            return int(idx[0]), int(idx[0])
        a, b = int(idx[0]), int(idx[1])
        if (a + 1) % n == b:
            return a, b
        if (b + 1) % n == a:
            return b, a
        return None

    left, right = run(at_min), run(at_max)
    if left is None or right is None:
        return None

    # Chain 1: from the end of the left run to the start of the right run; chain 2: from
    # the end of the right run back to the start of the left run.
    def chain(a: int, b: int) -> list[int]:
        out = [a]
        k = a
        while k != b:
            k = (k + 1) % n
            out.append(k)
        return out

    c1 = chain(left[1], right[0])
    c2 = chain(right[1], left[0])
    if not (np.all(np.diff(u[c1]) > 0) and np.all(np.diff(u[c2]) < 0)):
        return None
    c2 = c2[::-1]
    v1, v2 = float(np.mean(uv[c1, 1])), float(np.mean(uv[c2, 1]))
    bottom, top = (c1, c2) if v1 < v2 else (c2, c1)
    return bottom, top


def _rows(s, uv: np.ndarray, chord: float, angle: float):
    """Structured rows for a face of revolution bounded by two meridians (u = const) and
    two parallels (v = const, possibly poles): every meridian sample starts a row of
    points spaced for the row's own radius (the widest of its neighbours), and
    consecutive rows are zipped. Returns (uv points, triangles) or ``None``."""
    n = len(uv)
    u, v = uv[:, 0], uv[:, 1]
    span = float(u.max() - u.min())
    if n < 4 or span <= 0:
        return None
    eps = 1e-9 * max(span, 1.0)
    left = np.flatnonzero(u <= u.min() + eps)
    right = np.flatnonzero(u >= u.max() - eps)
    if len(left) < 2 or len(right) < 2:
        return None

    def contiguous(idx: np.ndarray) -> list[int] | None:
        # The run as a list in loop order (it may wrap around index 0).
        s_idx = set(idx.tolist())
        start = next((i for i in idx if (i - 1) % n not in s_idx), None)
        if start is None:
            return None
        out = [int(start)]
        while (out[-1] + 1) % n in s_idx:
            out.append((out[-1] + 1) % n)
        return out if len(out) == len(idx) else None

    lrun, rrun = contiguous(left), contiguous(right)
    if lrun is None or rrun is None:
        return None
    lchain = sorted(lrun, key=lambda i: v[i])
    rchain = sorted(rrun, key=lambda i: v[i])
    if len(lchain) != len(rchain) or np.max(np.abs(v[lchain] - v[rchain])) > 1e-9 * (
        abs(float(v.max() - v.min())) + 1.0
    ):
        return None
    # Bottom and top parallels: the remaining points between the meridians.
    used = set(lrun) | set(rrun)
    rest = [i for i in range(n) if i not in used]
    vb, vt = float(v[lchain[0]]), float(v[lchain[-1]])
    bottom_extra = sorted((i for i in rest if abs(v[i] - vb) <= 1e-9), key=lambda i: u[i])
    top_extra = sorted((i for i in rest if abs(v[i] - vt) <= 1e-9), key=lambda i: u[i])
    if len(bottom_extra) + len(top_extra) != len(rest):
        return None
    rows_idx: list[list[int]] = [[lchain[0], *bottom_extra, rchain[0]]]
    pts = [p for p in uv]
    m = len(lchain)
    vs = v[lchain]
    widths = []
    for k in range(m):
        su, _ = s.derivatives(np.linspace(u.min(), u.max(), 3), np.full(3, vs[k]))
        widths.append(float(np.max(np.linalg.norm(su, axis=1))))
    for k in range(1, m - 1):
        rho = max(widths[k - 1], widths[k], widths[k + 1], 1e-12)
        du = span if chord * 0.5 >= rho else min(2.0 * math.acos(1.0 - chord * 0.5 / rho), angle)
        count = int(min(max(math.ceil(span / du), 1), 4096))
        us = u.min() + span * np.arange(1, count) / count
        row = [lchain[k]]
        for uu in us:
            pts.append(np.array([uu, vs[k]]))
            row.append(len(pts) - 1)
        row.append(rchain[k])
        rows_idx.append(row)
    rows_idx.append([lchain[-1], *top_extra, rchain[-1]])
    allp = np.array(pts)
    tris = []
    for a, b in zip(rows_idx[:-1], rows_idx[1:], strict=True):
        tris += _zip(allp[:, 0], a, b)
    return allp, np.array(tris, dtype=np.int64)


def _doubly_curved_edges(body: Body) -> set[int]:
    """Edges bounding spheres, tori or revolution faces: sampled at half the chord, so
    that sag along both parameters together stays within the chord."""
    out: set[int] = set()
    for f in body.faces:
        if f.surface.kind in ("sphere", "torus", "revolution", "nurbs"):
            for lp in f.loops:
                out.update(ei for ei, _ in lp)
    return out


def _zip(u: np.ndarray, bottom: list[int], top: list[int]) -> list[tuple[int, int, int]]:
    """Triangles between two chains sorted by u (bottom below top)."""
    i = j = 0
    tris = []
    nb, nt = len(bottom), len(top)
    while i < nb - 1 or j < nt - 1:
        if j == nt - 1 or (i < nb - 1 and u[bottom[i + 1]] <= u[top[j + 1]]):
            tris.append((bottom[i], bottom[i + 1], top[j]))
            i += 1
        else:
            tris.append((bottom[i], top[j + 1], top[j]))
            j += 1
    return tris


def _radii(s, box) -> tuple[float, float]:
    """Smallest normal-curvature radii along u and v over a parameter box."""
    k = s.kind
    if k == "sphere":
        return s.radius, s.radius
    if k == "torus":
        return s.major - s.minor, s.minor
    u0, u1, v0, v1 = box
    us, vs = np.meshgrid(np.linspace(u0, u1, 9), np.linspace(v0, v1, 9), indexing="ij")
    us, vs = us.ravel(), vs.ravel()
    su, sv = s.derivatives(us, vs)
    suu, _, svv = s.second(us, vs)
    n = s.normal(us, vs)
    ku = np.abs(np.sum(suu * n, axis=1)) / np.maximum(np.sum(su * su, axis=1), 1e-300)
    kv = np.abs(np.sum(svv * n, axis=1)) / np.maximum(np.sum(sv * sv, axis=1), 1e-300)
    ru = 1.0 / max(float(ku.max()), 1e-12)
    rv = 1.0 / max(float(kv.max()), 1e-12)
    return ru, rv


def _step(radius: float, chord: float, angle: float) -> float:
    """Grid spacing for which an equilateral triangle deviates by about 0.6 ``chord``
    (circumradius rho with rho^2 / 2R = 0.6 chord; side sqrt(3) rho)."""
    if chord >= radius:
        return radius * angle
    return min(math.sqrt(3.0 * 1.2 * radius * chord), radius * angle)


def _seed(mesh: TriMesh2D, s, uv_loops, chord: float, angle: float) -> None:
    """Insert a curvature-adapted grid of interior points (rows in v, columns in u)."""
    allp = np.concatenate(uv_loops)
    u0, v0 = allp.min(axis=0)
    u1, v1 = allp.max(axis=0)
    ru, rv = _radii(s, (u0, u1, v0, v1))
    hu, hv = _step(ru, chord, angle), _step(rv, chord, angle)
    vs_probe = np.linspace(v0, v1, 9)
    us_probe = np.linspace(u0, u1, 9)
    uu, vv = np.meshgrid(us_probe, vs_probe, indexing="ij")
    _, sv = s.derivatives(uu.ravel(), vv.ravel())
    sv_max = max(float(np.max(np.linalg.norm(sv, axis=1))), 1e-12)
    # Staggered rows sqrt(3)/2 apart: near-equilateral triangles.
    nv = int(min(max(math.ceil((v1 - v0) * sv_max / (hv * 0.8660254)), 1), 2000))
    rows = v0 + (np.arange(1, nv) / nv) * (v1 - v0)
    cand = []
    for r, v in enumerate(rows):
        su, _ = s.derivatives(np.linspace(u0, u1, 5), np.full(5, v))
        su_max = max(float(np.max(np.linalg.norm(su, axis=1))), 1e-12)
        nu = int(min(max(math.ceil((u1 - u0) * su_max / hu), 1), 2000))
        shift = 0.5 if r % 2 else 0.0
        us = u0 + ((np.arange(1, nu) - shift) / nu) * (u1 - u0)
        cand.append(np.c_[us, np.full(len(us), v)])
    if not cand:
        return
    pts = np.concatenate(cand)
    if len(pts) == 0 or len(pts) > MAX_FACE_POINTS:
        return
    # Inside the domain (even-odd over every loop) and away from its boundary.
    inside = np.zeros(len(pts), dtype=bool)
    metric = np.array([mesh.sx, mesh.sy])
    far = np.ones(len(pts), dtype=bool)
    gap = 0.45 * min(hu, hv)
    for lp in uv_loops:
        a = lp
        b = np.roll(lp, -1, axis=0)
        cond = (a[:, 1][None] > pts[:, 1:2]) != (b[:, 1][None] > pts[:, 1:2])
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = (
                a[:, 0][None]
                + (pts[:, 1:2] - a[:, 1][None])
                * (b[:, 0] - a[:, 0])[None]
                / (b[:, 1] - a[:, 1])[None]
            )
        inside ^= (np.count_nonzero(cond & (pts[:, 0:1] < xc), axis=1) % 2).astype(bool)
        am, bm, pm = a * metric, b * metric, pts * metric
        d = bm - am
        dd = np.maximum(np.sum(d * d, axis=1), 1e-300)
        for chunk in range(0, len(pm), 2048):
            q = pm[chunk : chunk + 2048]
            t = np.clip(((q[:, None, :] - am[None]) * d[None]).sum(-1) / dd[None], 0, 1)
            proj = am[None] + t[..., None] * d[None]
            dist = np.min(np.linalg.norm(q[:, None, :] - proj, axis=-1), axis=1)
            far[chunk : chunk + 2048] &= dist > gap
    start = 0
    for p in pts[inside & far]:
        k = mesh.insert(float(p[0]), float(p[1]), start)
        if k >= 0:
            start = len(mesh.tri) - 1


def _refine(mesh: TriMesh2D, s, all_id, pos, nb: int, chord: float, angle: float) -> None:
    cos_angle = math.cos(angle)
    for _ in range(40):
        pts = mesh.points()
        tris = mesh.triangles()
        if len(pts) > MAX_FACE_POINTS:
            break
        uvc = pts[tris].mean(axis=1)
        p3 = _points3(s, pts, all_id, pos, nb)
        corners = p3[tris]
        centroid3 = corners.mean(axis=1)
        surf_c = s.point(uvc[:, 0], uvc[:, 1])
        dev = np.linalg.norm(surf_c - centroid3, axis=1)
        # Edge midpoints too: long edges of thin triangles sag the most.
        for a, b in ((0, 1), (1, 2), (2, 0)):
            uvm = 0.5 * (pts[tris[:, a]] + pts[tris[:, b]])
            pm = 0.5 * (corners[:, a] + corners[:, b])
            dev = np.maximum(dev, np.linalg.norm(s.point(uvm[:, 0], uvm[:, 1]) - pm, axis=1))
        nrm = s.normal(pts[:, 0], pts[:, 1])
        nt = nrm[tris]
        cmin = np.minimum(
            np.minimum(np.sum(nt[:, 0] * nt[:, 1], axis=1), np.sum(nt[:, 1] * nt[:, 2], axis=1)),
            np.sum(nt[:, 2] * nt[:, 0], axis=1),
        )
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 2] - corners[:, 0]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        # 0.9: a little margin over the centroid, the deepest point of well-shaped triangles.
        bad = np.flatnonzero(((dev > 0.9 * chord) | (cmin < cos_angle)) & (area > chord * chord))
        if not len(bad):
            break
        inserted = 0
        live = np.flatnonzero(mesh.alive)
        for k in bad:
            if mesh.insert(float(uvc[k, 0]), float(uvc[k, 1]), int(live[k])) >= 0:
                inserted += 1
        if not inserted:
            break


def _points3(s, pts, ids, pos, nb):
    out = np.empty((len(pts), 3))
    out[:nb] = pos[ids]
    if len(pts) > nb:
        out[nb:] = s.point(pts[nb:, 0], pts[nb:, 1])
    return out


def max_deviation(body: Body, tess: Tessellation, samples: int = 4) -> float:
    """Measured largest distance between triangle samples and their face's surface."""
    worst = 0.0
    bary = []
    for i in range(samples + 1):
        for j in range(samples + 1 - i):
            bary.append((i / samples, j / samples, 1 - (i + j) / samples))
    w = np.array(bary)
    for fi, face in enumerate(body.faces):
        mask = tess.face_of == fi
        if not np.any(mask):
            continue
        corners = tess.positions[tess.triangles[mask]]
        pts = np.einsum("bk,tkc->tbc", w, corners).reshape(-1, 3)
        if len(pts) > 20000:
            pts = pts[np.linspace(0, len(pts) - 1, 20000).astype(int)]
        d = (
            face.surface.distance(pts)
            if not isinstance(face.surface, Plane)
            else np.abs(face.surface.signed_distance(pts))
        )
        worst = max(worst, float(np.max(d)))
    return worst
