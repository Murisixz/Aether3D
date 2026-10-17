"""Solid booleans: Manifold decides the topology, the B-rep keeps the exact geometry.

``boolean(a, b, op)`` with ``op`` in ``"union"``, ``"difference"``, ``"intersection"``:

1. Both bodies are tessellated watertight at the boolean tolerance (``BOOLEAN``: about
   1e-4 of their size), every triangle tagged with the face it approximates.
2. Manifold (``manifold3d``, Apache-2.0; float64 meshes) computes the boolean of the two
   meshes; its result is always a closed 2-manifold and every output triangle keeps the
   id of the input triangle it came from (one id per input triangle).
3. The B-rep is **reconstructed**: output triangles are grouped by their source face into
   connected regions -- one face each, with the source's exact surface and its sense
   (flipped where the result shows the other side, as in a difference); coplanar or
   co-surface faces of *different* operands that touch are merged. Region boundaries are
   chained into edges between pairs of faces; every edge's curve is *recovered*: an
   original edge of an operand when the chain lies on it, else the closed-form
   intersection of the two surfaces (``intersect.recover_curve``: lines, circles,
   ellipses), else a B-spline traced on both surfaces with its measured deviation as the
   edge's tolerance. Vertices are refined onto all their faces' surfaces.
4. The result is validated (``topology.validate``); a result that does not validate is an
   error (``BOOLEAN_FAILED``), never returned.

What is exact: faces lie on their operands' exact surfaces; edges are analytic where
their surface pair has a closed form. What is approximate: topology decisions were made
on tessellations, so features thinner than the boolean tolerance (near tangencies, slivers
of a few micrometres) may be lost or reported (``NEAR_DEGENERATE``); traced edges carry
their measured deviation. The result's ``notes`` say which edges were traced.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ...errors import CadError
from ..geom3d import Plane
from ..tolerance import TessellationTolerance
from .intersect import recover_curve, trace
from .tessellate import tessellate
from .topology import Body, Edge, Face, edge_range, validate

OPS = ("union", "difference", "intersection")
BOOLEAN = TessellationTolerance(chord=1e-3, angle=math.radians(8.0), relative=1e-4)


def _manifold():
    try:
        import manifold3d
    except ImportError:
        raise CadError(
            "BOOLEAN_UNAVAILABLE",
            "solid booleans need the manifold3d package",
            hint='install it with: pip install "aether3d[boolean]"',
        ) from None
    return manifold3d


def _to_manifold(m3d, tess, face_base: int):
    gl = m3d.Mesh64(
        vert_properties=np.ascontiguousarray(tess.positions, dtype=np.float64),
        tri_verts=np.ascontiguousarray(tess.triangles, dtype=np.uint64),
        face_id=np.arange(face_base, face_base + len(tess.triangles), dtype=np.uint64),
    )
    m = m3d.Manifold(gl)
    status = m.status()
    if str(status).split(".")[-1] != "NoError":
        raise CadError(
            "BOOLEAN_FAILED",
            f"an operand's tessellation is not a closed manifold ({status})",
            hint="booleans need closed, valid solids",
        )
    return m


def boolean(a: Body, b: Body, op: str, *, tolerance: TessellationTolerance | None = None) -> Body:
    if op not in OPS:
        raise CadError("INVALID_OPERATION", f"op must be one of {OPS}")
    for name, body in (("first", a), ("second", b)):
        if body.kind != "solid":
            raise CadError("NOT_A_SOLID", f"the {name} operand is not a solid")
    m3d = _manifold()
    tol = tolerance or BOOLEAN
    size = max(a.size, b.size)
    ta = tessellate(a, tol)
    tb = tessellate(b, tol)
    ma = _to_manifold(m3d, ta, 0)
    mb = _to_manifold(m3d, tb, len(ta.triangles))
    if op == "union":
        res = ma + mb
    elif op == "difference":
        res = ma - mb
    else:
        res = ma ^ mb
    out = res.to_mesh64()
    verts = np.asarray(out.vert_properties, dtype=np.float64)[:, :3]
    tris = np.asarray(out.tri_verts, dtype=np.int64).reshape(-1, 3)
    fid = np.asarray(out.face_id, dtype=np.int64)
    if len(tris) == 0:
        raise CadError("EMPTY_RESULT", f"the {op} of the two solids is empty")
    na = len(ta.triangles)
    from_b = fid >= na
    src = np.where(from_b, tb.face_of[np.clip(fid - na, 0, None)] + len(a.faces), 0)
    src = np.where(from_b, src, ta.face_of[np.clip(fid, 0, na - 1)])
    sources = list(a.faces) + list(b.faces)
    owner = [0] * len(a.faces) + [1] * len(b.faces)
    chord = max(ta.chord, tb.chord)
    return reconstruct(verts, tris, src, sources, owner, (a, b), chord, size)


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def same_surface(s1, s2, tol: float) -> bool:
    """Whether two surfaces are the same point set (within ``tol``)."""
    if s1 is s2:
        return True
    if s1.kind != s2.kind:
        return False
    if s1.kind == "plane":
        n1, n2 = s1.frame.z, s2.frame.z
        if abs(abs(float(n1 @ n2)) - 1.0) > 1e-10:
            return False
        return abs(float(n1 @ (s2.frame.origin - s1.frame.origin))) <= tol
    if s1.kind in ("cylinder", "cone", "sphere", "torus"):
        if s1.kind == "sphere":
            return (
                float(np.linalg.norm(s1.frame.origin - s2.frame.origin)) <= tol
                and abs(s1.radius - s2.radius) <= tol
            )
        a1, a2 = s1.frame.z, s2.frame.z
        if abs(abs(float(a1 @ a2)) - 1.0) > 1e-10:
            return False
        if float(np.linalg.norm(np.cross(s2.frame.origin - s1.frame.origin, a1))) > tol:
            return False
        if s1.kind == "cylinder":
            return abs(s1.radius - s2.radius) <= tol
        if s1.kind == "torus":
            return (
                abs(s1.major - s2.major) <= tol
                and abs(s1.minor - s2.minor) <= tol
                and abs(float(a1 @ (s2.frame.origin - s1.frame.origin))) <= tol
            )
        # Cones: same apex and half angle (up to orientation).
        apex1 = s1.frame.origin + s1.apex_v * a1
        apex2 = s2.frame.origin + s2.apex_v * a2
        same_dir = float(a1 @ a2) > 0
        ang1 = s1.half_angle if same_dir else -s1.half_angle
        return float(np.linalg.norm(apex1 - apex2)) <= tol and abs(ang1 - s2.half_angle) <= 1e-10
    return False


def _outward(face: Face, point: np.ndarray) -> np.ndarray:
    s = face.surface
    uv = s.inverse(point[None])[0]
    n = s.normal(uv[0], uv[1]).reshape(3)
    return n if face.sense else -n


def _refine_vertex(surfaces: list, p: np.ndarray) -> np.ndarray:
    q = np.array(p, dtype=np.float64)
    for _ in range(25):
        rows, rhs = [], []
        for s in surfaces:
            if isinstance(s, Plane):
                n = s.frame.z
                rows.append(n)
                rhs.append(float(n @ (q - s.frame.origin)))
            else:
                uv = s.inverse(q[None])[0]
                f = s.point(uv[0], uv[1]).reshape(3)
                n = s.normal(uv[0], uv[1]).reshape(3)
                rows.append(n)
                rhs.append(float(n @ (q - f)))
        a = np.array(rows)
        step = np.linalg.lstsq(a, np.array(rhs), rcond=None)[0]
        q = q - step
        if float(np.linalg.norm(step)) < 1e-15 * max(1.0, float(np.max(np.abs(q)))):
            break
    return q


def reconstruct(verts, tris, src, sources, owner, bodies, chord, size) -> Body:
    nt = len(tris)
    tol_fit = max(6.0 * chord, 1e-9 * size)
    merge_tol = max(1e-9 * size, 1e-12)
    # Half-edge adjacency.
    he: dict[tuple[int, int], int] = {}
    for t in range(nt):
        a, b, c = (int(x) for x in tris[t])
        he[(a, b)] = t
        he[(b, c)] = t
        he[(c, a)] = t
    # Regions: triangles of one source face connected through shared edges.
    uf = _UF(nt)
    for (a, b), t in he.items():
        o = he.get((b, a))
        if o is not None and src[o] == src[t]:
            uf.union(t, o)
    region_of_tri = np.array([uf.find(t) for t in range(nt)])
    roots, region = np.unique(region_of_tri, return_inverse=True)
    nr = len(roots)
    reg_src = np.zeros(nr, dtype=np.int64)
    reg_src[region] = src
    # Merge touching regions of different operands lying on the same surface (with the
    # same outward side).
    tri_n = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    tri_area = 0.5 * np.linalg.norm(tri_n, axis=1)
    big = np.zeros(nr, dtype=np.int64)
    best = np.full(nr, -1.0)
    for t in range(nt):
        r = region[t]
        if tri_area[t] > best[r]:
            best[r] = tri_area[t]
            big[r] = t
    centroid = verts[tris].mean(axis=1)
    reg_areas = np.bincount(region, weights=tri_area, minlength=nr)
    reg_flip = np.zeros(nr, dtype=bool)
    for r in range(nr):
        t = big[r]
        n = tri_n[t]
        out = _outward(sources[reg_src[r]], centroid[t])
        reg_flip[r] = float(n @ out) < 0
    ruf = _UF(nr)
    pairs = set()
    for (a, b), t in he.items():
        o = he.get((b, a))
        if o is None:
            continue
        r1, r2 = region[t], region[o]
        if r1 != r2:
            pairs.add((min(r1, r2), max(r1, r2)))
    for r1, r2 in sorted(pairs):
        s1, s2 = reg_src[r1], reg_src[r2]
        f1, f2 = sources[s1], sources[s2]
        # Faces of one operand on one periodic surface stay apart (an operand's
        # cylinder halves would merge into a band without seams); coplanar faces merge.
        if owner[s1] == owner[s2] and f1.surface.kind != "plane":
            continue
        if same_surface(f1.surface, f2.surface, merge_tol):
            n1 = tri_n[big[r1]]
            n2 = tri_n[big[r2]]
            # Same orientation, or a sliver (coplanar faces of both operands leave
            # regions of rounding-size area whose normals mean nothing).
            sliver = min(reg_areas[r1], reg_areas[r2]) <= (1e-7 * size) ** 2
            if float(n1 @ n2) > 0 or sliver:
                ruf.union(r1, r2)
    face_of_region = np.array([ruf.find(r) for r in range(nr)])
    fids, face_index = np.unique(face_of_region, return_inverse=True)
    nf = len(fids)
    face_of_tri = face_index[region]
    # Representative source of each face: its largest region.
    rep_src = np.zeros(nf, dtype=np.int64)
    rep_area = np.full(nf, -1.0)
    rep_flip = np.zeros(nf, dtype=bool)
    for r in range(nr):
        f = face_index[r]
        if reg_areas[r] > rep_area[f]:
            rep_area[f] = reg_areas[r]
            rep_src[f] = reg_src[r]
            rep_flip[f] = reg_flip[r]
    # Boundary half-edges of each face and their neighbours.
    boundary: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    neighbour: dict[tuple[int, int], int] = {}
    for (a, b), t in he.items():
        o = he.get((b, a))
        if o is None:
            raise CadError("BOOLEAN_FAILED", "the boolean result is not closed")
        f, g = face_of_tri[t], face_of_tri[o]
        if f != g:
            boundary[f][a].append((a, b))
            neighbour[(a, b)] = g
    # Loops per face, then chains by neighbour.
    vertex_faces: dict[int, set[int]] = defaultdict(set)
    for t in range(nt):
        for v in tris[t]:
            vertex_faces[int(v)].add(int(face_of_tri[t]))
    face_loops: dict[int, list[list[tuple[int, int]]]] = {}
    for f, outgoing in boundary.items():
        remaining = {k: list(v) for k, v in outgoing.items()}
        loops = []
        for start in sorted(remaining):
            while remaining.get(start):
                loop = []
                cur = remaining[start].pop()
                loop.append(cur)
                guard = 0
                while cur[1] != loop[0][0]:
                    nxt = remaining.get(cur[1])
                    if not nxt:
                        raise CadError("BOOLEAN_FAILED", "a face boundary of the result is open")
                    if len(nxt) > 1:
                        # Pinch vertex: continue with the same neighbour when possible.
                        same = [h for h in nxt if neighbour[h] == neighbour[cur]]
                        cur = same[0] if same else nxt[0]
                        nxt.remove(cur)
                    else:
                        cur = nxt.pop()
                    loop.append(cur)
                    guard += 1
                    if guard > len(neighbour):
                        raise CadError("BOOLEAN_FAILED", "could not chain a face boundary")
                loops.append(loop)
        face_loops[f] = loops
    # Junction vertices: where the neighbour changes along a loop, or where the vertex
    # touches three or more faces.
    junction: set[int] = set()
    for loops in face_loops.values():
        for loop in loops:
            n = len(loop)
            for i in range(n):
                if neighbour[loop[i]] != neighbour[loop[i - 1]]:
                    junction.add(loop[i][0])
    for v, fs in vertex_faces.items():
        if len(fs) >= 3:
            junction.add(v)
    # Vertices of the operands that survive stay vertices (a cone's apex joins only its
    # two halves, but the seams on either side of it are separate edges).
    corners = np.concatenate([b.vertices for b in bodies if len(b.vertices)])
    if len(corners):
        cand = np.array(sorted({v for loops in face_loops.values() for lp in loops for v, _ in lp}))
        if len(cand):
            pos = verts[cand]
            for c in corners:
                d = np.max(np.abs(pos - c), axis=1)
                for k in np.flatnonzero(d <= merge_tol):
                    junction.add(int(cand[k]))
    # Loops with fewer than two junctions (a hole's rim) get canonical split vertices --
    # chosen from the vertex set alone (the lowest vertex and the vertex farthest from it),
    # so the faces on both sides choose the same ones.
    for loops in face_loops.values():
        for loop in loops:
            vs = [h[0] for h in loop]
            have = [v for v in vs if v in junction]
            if len(have) >= 2:
                continue
            first = have[0] if have else min(vs, key=lambda v: tuple(verts[v]))
            d = np.linalg.norm(verts[vs] - verts[first], axis=1)
            far = max(range(len(vs)), key=lambda i: (round(float(d[i]), 12), tuple(-verts[vs[i]])))
            junction.add(first)
            junction.add(vs[far])
    # Edges: chains between junctions, shared by both faces.
    surfaces = [sources[rep_src[f]].surface for f in range(nf)]
    vmap: dict[int, int] = {}
    vpos: list[np.ndarray] = []

    def bvertex(v: int) -> int:
        if v not in vmap:
            fs = sorted(vertex_faces[v])
            p = _refine_vertex([surfaces[f] for f in fs], verts[v])
            vmap[v] = len(vpos)
            vpos.append(p)
        return vmap[v]

    edges: list[Edge] = []
    edge_key: dict[tuple, int] = {}
    traced = 0
    max_dev = 0.0
    origin_edges = _operand_edges(bodies, sources, owner)
    face_coedges: dict[int, list[list[tuple[int, bool]]]] = {}
    for f in sorted(face_loops):
        coloops = []
        for loop in face_loops[f]:
            n = len(loop)
            start = next(i for i in range(n) if loop[i][0] in junction)
            chains: list[list[tuple[int, int]]] = []
            cur: list[tuple[int, int]] = []
            for k in range(n):
                h = loop[(start + k) % n]
                if cur and h[0] in junction:
                    chains.append(cur)
                    cur = []
                cur.append(h)
            chains.append(cur)
            co = []
            for ch in chains:
                g = neighbour[ch[0]]
                vs = [h[0] for h in ch] + [ch[-1][1]]
                canon = vs if f < g else vs[::-1]
                key = (min(f, g), max(f, g), tuple(canon))
                if key not in edge_key:
                    pts = verts[canon]
                    v0, v1 = bvertex(canon[0]), bvertex(canon[-1])
                    s1, s2 = surfaces[min(f, g)], surfaces[max(f, g)]
                    curve = _original(origin_edges, rep_src, min(f, g), max(f, g), pts, tol_fit)
                    etol = 0.0
                    if curve is None:
                        curve = recover_curve(s1, s2, pts, tol_fit)
                    if curve is None:
                        curve, etol = trace(s1, s2, pts)
                        traced += 1
                        max_dev = max(max_dev, etol)
                    p0, p1 = vpos[v0], vpos[v1]
                    pm = pts[len(pts) // 2]
                    t0, t1 = edge_range(curve, p0, pm, p1)
                    q0 = curve.point(t0)
                    if np.linalg.norm(q0 - p0) <= np.linalg.norm(q0 - p1):
                        e = Edge(curve, t0, t1, v0, v1, "", etol)
                        fwd_canon = True
                    else:
                        e = Edge(curve, t0, t1, v1, v0, "", etol)
                        fwd_canon = False
                    edge_key[key] = len(edges)
                    edges.append(e)
                    edge_dir = fwd_canon
                else:
                    e = edges[edge_key[key]]
                    edge_dir = e.v0 == vmap[canon[0]]
                ei = edge_key[key]
                forward = edge_dir if f < g else not edge_dir
                co.append((ei, forward))
            coloops.append(co)
        face_coedges[f] = coloops
    # Faces with their loops ordered (outer first), names from their sources.
    faces: list[Face] = []
    name_count: dict[str, int] = defaultdict(int)
    for f in range(nf):
        name_count[sources[rep_src[f]].name] += 1
    order = sorted(
        range(nf), key=lambda f: tuple(np.round(centroid[face_of_tri == f].mean(axis=0), 9))
    )
    seen: dict[str, int] = defaultdict(int)
    names = {}
    for f in order:
        base = sources[rep_src[f]].name
        if name_count[base] > 1:
            names[f] = f"{base}#{seen[base]}"
            seen[base] += 1
        else:
            names[f] = base
    vpos_arr = np.array(vpos)
    for f in range(nf):
        sf = sources[rep_src[f]]
        sense = sf.sense if not rep_flip[f] else not sf.sense
        loops = face_coedges.get(f, [])
        face = Face(
            sf.surface, sense, tuple(tuple(lp) for lp in loops), names[f], sf.group, sf.window
        )
        faces.append(_outer_first(face, edges, vpos_arr))
    # Shells: faces connected through edges; outer shells first.
    suf = _UF(nf)
    for (a, b), t in he.items():
        o = he[(b, a)]
        suf.union(int(face_of_tri[t]), int(face_of_tri[o]))
    groups: dict[int, list[int]] = defaultdict(list)
    for f in range(nf):
        groups[suf.find(f)].append(f)
    shell_vol = {}
    for root, fs in groups.items():
        mask = np.isin(face_of_tri, fs)
        p = verts[tris[mask]]
        shell_vol[root] = (
            float(np.sum(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])))) / 6
        )
    shells = sorted(groups.values(), key=lambda fs: -shell_vol[suf.find(fs[0])])
    vtol = np.zeros(len(vpos))
    for e in edges:
        for v, t in ((e.v0, e.t0), (e.v1, e.t1)):
            vtol[v] = max(vtol[v], float(np.linalg.norm(e.curve.point(t) - vpos_arr[v])))
    notes = []
    if traced:
        notes.append(
            f"{traced} intersection edge(s) traced numerically (max deviation {max_dev:.3g} m)"
        )
    body = Body(
        vpos_arr,
        tuple(edges),
        tuple(faces),
        tuple(tuple(s) for s in shells),
        "solid",
        vtol,
        tuple(notes),
    )
    report = validate(body)
    if not report["valid"]:
        raise CadError(
            "BOOLEAN_FAILED",
            "the boolean result could not be reconstructed as a valid B-rep",
            details={"problems": report["problems"][:10]},
        )
    return body


def _outer_first(face: Face, edges: list[Edge], vpos: np.ndarray) -> Face:
    if len(face.loops) <= 1:
        return face
    s = face.surface
    areas = []
    for loop in face.loops:
        pts = []
        for ei, fwd in loop:
            e = edges[ei]
            p = e.curve.point(np.linspace(e.t0, e.t1, 9))
            pts.append(p if fwd else p[::-1])
        p = np.concatenate(pts)
        uv = (
            face.to_window(s.inverse(p)) if not isinstance(s, Plane) else s.frame.to_local(p)[:, :2]
        )
        x, y = uv[:, 0], uv[:, 1]
        a = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        areas.append(a if face.sense else -a)
    order = sorted(range(len(face.loops)), key=lambda i: -areas[i])
    return Face(
        face.surface,
        face.sense,
        tuple(face.loops[i] for i in order),
        face.name,
        face.group,
        face.window,
    )


def _operand_edges(bodies, sources, owner):
    """For every pair of source faces adjacent in an operand: the operand's edges."""
    out: dict[tuple[int, int], list[Edge]] = defaultdict(list)
    base = 0
    for body in bodies:
        uses = body.edge_faces()
        for ei, fs in uses.items():
            if len(fs) == 2:
                f1, f2 = fs[0][0] + base, fs[1][0] + base
                out[(min(f1, f2), max(f1, f2))].append(body.edges[ei])
        base += len(body.faces)
    return out


def _original(origin_edges, rep_src, f: int, g: int, pts: np.ndarray, tol: float):
    s1, s2 = int(rep_src[f]), int(rep_src[g])
    for e in origin_edges.get((min(s1, s2), max(s1, s2)), ()):
        t = e.curve.inverse(pts)
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        q = e.curve.point(t)
        if float(np.max(np.linalg.norm(q - pts, axis=1))) <= tol:
            return e.curve
    return None
