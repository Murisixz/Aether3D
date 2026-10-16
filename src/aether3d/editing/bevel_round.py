"""The general bevel: several segments, superellipse profiles, vertices on open boundaries.

:meth:`BevelMixin.bevel_edges <aether3d.editing.bevel.BevelMixin.bevel_edges>` uses this
module for every bevel except a single-segment one whose vertices all have closed fans
(that case keeps the chamfer algorithm of Aether3D 0.3-0.12 unchanged).

Geometry
--------
The faces next to a bevelled edge shrink exactly as for a chamfer: in each, the new
boundary runs parallel to the edge at distance ``width`` (mitres where two bevelled edges
meet inside a face, slides along unbevelled edges). The chamfer face of each edge becomes
a **strip** of ``segments`` faces between two **profiles**, one at each end of the edge.

A profile at vertex ``v`` runs from the edge's point in its first face (``start``) to its
point in its second face (``end``). It is a superellipse quadrant mapped affinely onto the
parallelogram with corners ``start``, ``middle`` and ``end``: the point ``(x, y)`` of
``x^r + y^r = 1`` maps to ``O + x (middle - start) + y (middle - end)`` with
``O = start + end - middle``, so ``(0, 1)`` is ``start``, ``(1, 0)`` is ``end`` and the
square's corner ``(1, 1)`` would be ``middle``. ``r = -2 ln 2 / ln(profile)`` (Blender's
mapping): ``profile = 0.5`` gives ``r = 2``, an elliptical arc that is circular where the
two faces meet at a right angle and the widths agree; ``0.25`` a straight chamfer. Points
are spaced evenly in angle on the circle and by arc length otherwise.

``middle`` is where the two faces' tangent lines meet, i.e. where the edge's line crosses
the profile's plane. If either end of the profile slides along an unbevelled edge, the
plane contains ``v`` and ``middle`` is ``v`` (the end face of a single bevelled edge; the
mitred end of a bevelled path, where it gives the exact intersection of the fillet with
the mitre plane). If both ends are mitre points (a vertex all of whose nearby edges are
bevelled), the plane is the cross-section of the edge (as perpendicular to it as the two
points allow), so the strip is a true fillet up to the corner patch.

Corners
-------
Around each vertex, the new points of its faces and the profiles' inner points form the
corner's outline, which closes in one of these ways (in this order):

1. it folds back on itself (two strips sharing a profile, a bevelled path): nothing;
2. it is one profile and its chord, and a face already runs along the chord: the face
   takes the profile (it lies in that face's plane);
3. it has zero area with every point on one line (a flat vertex): the edges on both sides
   of the line are subdivided by each other's points (no T-junction), as the chamfer
   merges such corners into the chamfer face;
4. it is planar and simple: one polygon;
5. every edge of the vertex is bevelled and ``segments >= 2``: a grid patch (rings of
   quads around a center vertex, or a center polygon for odd ``segments``); interior
   points interpolate the outline with mean value coordinates (Floater 2003) plus a bulge
   toward the profiles' bulge, and are projected onto the sphere through the outline when
   the outline lies on one (a box corner with circular profiles is an exact octant);
6. otherwise: a fan of triangles around a center vertex.

At an open boundary the outline is an open chain; it is closed along the boundary (case 4
or 6) only if it covers part of the original surface (it contains a point that is not on
a profile); a corner made only of a profile stays open, so the boundary follows the arc.

Attributes: rebuilt faces take an affine fit over the original face (exact for affine
UVs on planar faces); strip corners interpolate between the two faces' values along the
profile; patch and fan points interpolate the outline's values with the same weights as
their positions.
"""

from __future__ import annotations

import math

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import EditError
from . import _polygon as poly
from .core import Selection

MAX_SEGMENTS = 64
"""Most segments a bevel may have."""

_DENSE = 1024  # samples for arc-length spacing of non-circular superellipses


def superellipse_r(profile: float) -> float:
    """The superellipse exponent of a profile shape (Blender's mapping)."""
    return -2.0 * math.log(2.0) / math.log(profile)


def _quadrant(n: int, r: float) -> np.ndarray:
    """``n + 1`` points of ``x^r + y^r = 1`` from ``(0, 1)`` to ``(1, 0)``."""
    if abs(r - 2.0) <= 1e-12:
        theta = (math.pi / 2) * (1.0 - np.arange(n + 1) / n)
        pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    else:
        theta = (math.pi / 2) * (1.0 - np.arange(_DENSE + 1) / _DENSE)
        c, s = np.clip(np.cos(theta), 0.0, 1.0), np.clip(np.sin(theta), 0.0, 1.0)
        dense = np.stack([c ** (2.0 / r), s ** (2.0 / r)], axis=1)
        steps = np.sqrt((np.diff(dense, axis=0) ** 2).sum(axis=1))
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        target = arc[-1] * np.arange(n + 1) / n
        pts = np.stack(
            [np.interp(target, arc, dense[:, 0]), np.interp(target, arc, dense[:, 1])], axis=1
        )
    pts[0] = (0.0, 1.0)
    pts[-1] = (1.0, 0.0)
    return pts


def profile_points(
    start: np.ndarray, middle: np.ndarray, end: np.ndarray, n: int, r: float
) -> np.ndarray:
    """``n + 1`` points of the profile from ``start`` to ``end`` with corner ``middle``."""
    q = _quadrant(n, r)
    origin = start + end - middle
    u, v = middle - start, middle - end
    out = origin + q[:, :1] * u + q[:, 1:] * v
    out[0], out[-1] = start, end
    return out


def _collapse_spikes(seq: list[tuple[int, int]], closed: bool) -> list[tuple[int, int]]:
    """Drop repeated points and back-and-forth spikes (``x, y, x`` -> ``x``)."""
    out = list(seq)
    changed = True
    while changed and out:
        changed = False
        dedup: list[tuple[int, int]] = []
        for item in out:
            if dedup and dedup[-1][0] == item[0]:
                changed = True
                continue
            dedup.append(item)
        if closed and len(dedup) > 1 and dedup[0][0] == dedup[-1][0]:
            dedup.pop()
            changed = True
        out = dedup
        m = len(out)
        if m >= 3:
            for i in range(m if closed else m - 2):
                a, b, c = out[i][0], out[(i + 1) % m][0], out[(i + 2) % m][0]
                if a == c and a != b:
                    j = (i + 1) % m
                    k = (i + 2) % m
                    out = [x for idx, x in enumerate(out) if idx not in (j, k)]
                    changed = True
                    break
        elif closed and m == 2:
            out = []
    return out


def _split_at_repeats(hole: list[int]) -> list[list[int]]:
    """Split a closed outline that visits a vertex twice into simple outlines."""
    stack, done = [hole], []
    while stack:
        h = stack.pop()
        seen: dict[int, int] = {}
        for i, x in enumerate(h):
            if x in seen:
                j = seen[x]
                stack.append(h[j:i])
                stack.append(h[:j] + h[i:])
                break
            seen[x] = i
        else:
            done.append(h)
    return [h for h in done if len(h) >= 3]


def _mean_value_weights(domain: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Mean value coordinates (Floater 2003) of interior ``points`` w.r.t. polygon ``domain``."""
    d = domain[None, :, :] - points[:, None, :]  # (p, k, 2)
    dist = np.sqrt((d**2).sum(axis=2))
    nxt = np.roll(d, -1, axis=1)
    cross = d[..., 0] * nxt[..., 1] - d[..., 1] * nxt[..., 0]
    dot = (d * nxt).sum(axis=2)
    alpha = np.arctan2(cross, dot)  # angle at the point between vertex i and i + 1
    t = np.tan(alpha / 2.0)
    w = (np.roll(t, 1, axis=1) + t) / dist
    return w / w.sum(axis=1, keepdims=True)


def _fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Least-squares sphere through ``points``: ``(center, radius, max deviation)``."""
    if len(points) < 4:
        return None
    a = np.c_[2.0 * points, np.ones(len(points))]
    b = (points**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = sol[:3]
    r2 = sol[3] + center @ center
    if not r2 > 0:
        return None
    radius = math.sqrt(r2)
    dev = float(np.abs(np.sqrt(((points - center) ** 2).sum(axis=1)) - radius).max())
    return center, radius, dev


def _plane_deviation(points: np.ndarray) -> float:
    """Largest distance of ``points`` from their Newell plane (inf if there is none)."""
    nrm = poly.unit(poly.newell_normal(points))
    if nrm is None:
        return math.inf
    c = points.mean(axis=0)
    return float(np.abs((points - c) @ nrm).max())


class RoundBevelMixin:
    def _vertex_fan(self, v: int) -> tuple[list[tuple[int, int, int, int, int]], bool]:
        """Faces around ``v`` in order as ``(face, edge_in, edge_out, p, q)``, and whether
        the fan is closed. ``face`` runs ``p -> v -> q``; consecutive entries share
        ``edge_out``/``edge_in``. An open fan starts at the face whose ``edge_in`` is on the
        boundary and ends at the face whose ``edge_out`` is. Requires one manifold,
        consistently wound fan."""
        edges = self._vert_edges[v]
        radial = {e: len(self._edge_faces[e]) for e in edges}
        boundary = [e for e, k in radial.items() if k == 1]
        if any(k > 2 for k in radial.values()):
            raise EditError(
                "BEVEL_UNSUPPORTED_VERTEX",
                f"vertex {v} is on a non-manifold edge",
                elements={"vertices": [v]},
            )
        if not boundary:
            return self._closed_fan(v), True
        if len(boundary) != 2 or len(edges) < 3:
            raise EditError(
                "BEVEL_UNSUPPORTED_VERTEX",
                f"vertex {v} has {len(boundary)} boundary edges and {len(edges)} edges; "
                "a bevelled vertex on a boundary needs one open fan of at least two faces",
                elements={"vertices": [v]},
            )
        faces = self._faces_of_vertex(v)
        start = None
        for f in sorted(faces):
            loop = self._faces[f]
            k = loop.index(v)
            if self._edge_id(loop[k - 1], v) in boundary:
                start = f
                break
        if start is None:
            raise EditError(
                "BEVEL_UNSUPPORTED_VERTEX",
                f"faces around vertex {v} are wound inconsistently",
                elements={"vertices": [v]},
            )
        fan = []
        f = start
        while True:
            loop = self._faces[f]
            k = loop.index(v)
            p, q = loop[k - 1], loop[(k + 1) % len(loop)]
            e_in, e_out = self._edge_id(p, v), self._edge_id(v, q)
            fan.append((f, e_in, e_out, p, q))
            if e_out in boundary or len(fan) > len(faces):
                break
            g_faces = self._edge_faces[e_out]
            g = g_faces[0] if g_faces[1] == f else g_faces[1]
            lg = self._faces[g]
            if lg[lg.index(v) - 1] != q:
                raise EditError(
                    "BEVEL_UNSUPPORTED_VERTEX",
                    f"faces around vertex {v} are wound inconsistently",
                    elements={"vertices": [v]},
                )
            f = g
        if len(fan) != len(faces) or fan[-1][2] not in boundary:
            raise EditError(
                "BEVEL_UNSUPPORTED_VERTEX",
                f"vertex {v} is non-manifold (its faces form more than one fan)",
                elements={"vertices": [v]},
            )
        return fan, False

    def _bevel_general(self, eids: list[int], w: float, n: int, shape: float, fans: dict):
        r = superellipse_r(shape)
        bevelled = set(eids)
        vertices = sorted({v for e in eids for v in self._edge_verts[e]})
        fan_of = {v: fans[v][0] for v in vertices}
        closed_at = {v: fans[v][1] for v in vertices}
        normal_cache: dict[int, np.ndarray] = {}
        scale = self._scale()

        def normal(f: int) -> np.ndarray:
            if f not in normal_cache:
                nf = self._face_normal_vec(f)
                if nf is None:
                    raise EditError(
                        "DEGENERATE_INPUT", f"face {f} has zero area", elements={"faces": [f]}
                    )
                normal_cache[f] = nf
            return normal_cache[f]

        def inward(f: int, a: int, b: int) -> np.ndarray:
            d = poly.unit(poly.cross3(normal(f), self._pos[b] - self._pos[a]))
            if d is None:
                raise EditError(
                    "DEGENERATE_INPUT", f"face {f} has a zero-length edge", elements={"faces": [f]}
                )
            return d

        # 1. Point requests: mitres and slides (as the chamfer).
        slide_requests: dict[tuple[int, int], list[np.ndarray]] = {}
        mitre: dict[tuple[int, int], np.ndarray] = {}
        for v in vertices:
            pv = self._pos[v]
            for f, e_in, e_out, p, q in fan_of[v]:
                b_in, b_out = e_in in bevelled, e_out in bevelled
                if b_in and b_out:
                    d_in, d_out = inward(f, p, v), inward(f, v, q)
                    m = poly.unit(d_in + d_out)
                    if m is None or float(np.dot(m, d_in)) < 1e-3:
                        raise EditError(
                            "BEVEL_WIDTH_TOO_LARGE",
                            f"bevelled edges fold back at vertex {v}",
                            elements={"vertices": [v]},
                        )
                    mitre[(v, f)] = pv + w / float(np.dot(m, d_in)) * m
                elif b_in or b_out:
                    d = inward(f, p, v) if b_in else inward(f, v, q)
                    other, edge = (q, e_out) if b_in else (p, e_in)
                    u = poly.unit(self._pos[other] - pv)
                    c = float(np.dot(u, d)) if u is not None else 0.0
                    if c < 1e-3:
                        raise EditError(
                            "BEVEL_WIDTH_TOO_LARGE",
                            f"edge {edge} runs (almost) parallel to a "
                            f"bevelled edge at vertex {v}; the corner cannot slide along it",
                            elements={"edges": [edge]},
                        )
                    slide_requests.setdefault((v, edge), []).append(pv + w / c * u)
        slide: dict[tuple[int, int], np.ndarray] = {}
        uneven = []
        for key, points in slide_requests.items():
            slide[key] = np.mean(points, axis=0)
            if len(points) == 2 and np.linalg.norm(points[0] - points[1]) > 1e-9 * scale:
                uneven.append(key[1])
        if uneven:
            self._diagnose(
                Diagnostic(
                    "BEVEL_UNEVEN_WIDTH",
                    INFO,
                    "two faces requested different slide positions on an edge; their "
                    "average was used, so the width there is approximate",
                    None,
                    len(uneven),
                    tuple(sorted(uneven)[:10]),
                )
            )
        for (v, e), point in slide.items():
            a, b = self._edge_verts[e]
            other = b if a == v else a
            length = float(np.linalg.norm(self._pos[other] - self._pos[v]))
            used = float(np.linalg.norm(point - self._pos[v]))
            if (other, e) in slide:
                used += float(np.linalg.norm(slide[(other, e)] - self._pos[other]))
            if used >= length * (1 - 1e-9):
                raise EditError(
                    "BEVEL_WIDTH_TOO_LARGE",
                    f"width {w} is too large for edge {e}",
                    elements={"edges": [e]},
                    hint="use a smaller width",
                )

        # 2. Corner replacement lists (in each face's loop direction).
        point_id: dict[tuple, int] = {}
        new_vertices: list[int] = []
        mitre_points: set[int] = set()

        def vertex_for(key: tuple, position: np.ndarray) -> int:
            if key not in point_id:
                point_id[key] = self._new_vertex(position)
                new_vertices.append(point_id[key])
            return point_id[key]

        replacement: dict[tuple[int, int], list[int]] = {}
        retained: set[int] = set()
        for v in vertices:
            for f, e_in, e_out, _p, _q in fan_of[v]:
                if (v, f) in mitre:
                    x = vertex_for(("m", v, f), mitre[(v, f)])
                    mitre_points.add(x)
                    replacement[(v, f)] = [x]
                    continue
                seq = []
                for e in (e_in, e_out):
                    if e in bevelled:
                        continue
                    if (v, e) in slide:
                        seq.append(vertex_for(("s", v, e), slide[(v, e)]))
                    else:
                        seq.append(v)
                        retained.add(v)
                replacement[(v, f)] = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]

        # 3. Profiles at both ends of every bevelled edge; equal ends at one vertex share one.
        edge_faces = self._edge_faces_index(fan_of)
        directed = self._directed_index(fan_of)
        profiles: dict[tuple[int, int, int], list[int]] = {}
        for e in eids:
            f, g = edge_faces[e]
            a, b = directed[(e, f)]
            ends = (
                (a, self._point_in(a, f, replacement), self._point_in(a, g, replacement, False)),
                (b, self._point_in(b, f, replacement, False), self._point_in(b, g, replacement)),
            )
            for v, start, end in ends:
                key = (v, min(start, end), max(start, end))
                if key in profiles:
                    continue
                lo, hi = key[1], key[2]
                middle = self._profile_middle(v, e, lo, hi, mitre_points)
                pts = profile_points(self._pos[lo], middle, self._pos[hi], n, r)
                ids = [lo]
                for k in range(1, n):
                    # A straight profile through the vertex itself (a flat vertex) reuses it.
                    if np.linalg.norm(pts[k] - self._pos[v]) <= 1e-9 * scale and v not in ids:
                        ids.append(v)
                        retained.add(v)
                    else:
                        ids.append(vertex_for(("p", v, lo, hi, k), pts[k]))
                profiles[key] = [*ids, hi]

        def profile_path(v: int, start: int, end: int) -> list[int]:
            ids = profiles[(v, min(start, end), max(start, end))]
            return ids if ids[0] == start else ids[::-1]

        inner_points = {x for ids in profiles.values() for x in ids[1:-1]}
        profile_ends = {x for ids in profiles.values() for x in (ids[0], ids[-1])}

        # 4. Faces around the bevelled vertices, rebuilt (attributes later).
        affected = sorted({f for v in vertices for f, *_ in fan_of[v]})
        new_loops: dict[int, tuple[list[int], list[dict | None]]] = {}
        for f in affected:
            loop, corners = self._faces[f], self._corners_of(f)
            new_loop: list[int] = []
            new_corners: list[dict | None] = []
            for v, c in zip(loop, corners, strict=True):
                if (v, f) in replacement:
                    for x in replacement[(v, f)]:
                        new_loop.append(x)
                        new_corners.append(c if x == v else None)
                else:
                    new_loop.append(v)
                    new_corners.append(c)
            new_loops[f] = (new_loop, new_corners)
        original_normals = {f: normal(f) for f in affected}

        # 5. Corner outlines, and how each closes.
        outlines: list[tuple[int, list[int], dict[int, int]]] = []
        for v in vertices:
            fan, closed = fan_of[v], closed_at[v]
            seq: list[tuple[int, int]] = []
            for i, (f, _e_in, e_out, _p, _q) in enumerate(fan):
                seq += [(x, f) for x in replacement[(v, f)]]
                if e_out in bevelled and (closed or i + 1 < len(fan)):
                    g = fan[(i + 1) % len(fan)][0]
                    path = profile_path(v, replacement[(v, f)][-1], replacement[(v, g)][0])
                    seq += [(x, -1) for x in path[1:-1]]
            seq = _collapse_spikes(seq, closed)
            if len(seq) < 3:
                continue
            ids = [x for x, _ in seq][::-1]  # the faces run the fan forwards; corners back
            owner = {}
            for x, o in seq:
                owner.setdefault(x, o)
            if not closed and all(x in inner_points or x in profile_ends for x in ids):
                continue  # an open corner of profile points only: the boundary follows it
            for part in _split_at_repeats(ids):
                outlines.append((v, part, owner))

        merges: list[tuple[int, int, int, list[int]]] = []  # (face, x, y, points between)
        chains: list[tuple[int, list[int]]] = []
        fills: list[tuple[int, list[int], dict[int, int], str]] = []
        for v, hole, owner in outlines:
            merged = self._merge_into_face(hole, new_loops, inner_points)
            if merged is not None:
                merges.append(merged)
                continue
            pts = self._pos[hole]
            if poly.polygon_area(pts) <= self._area_eps():
                chain = self._collinear_chain(hole)
                if chain is None:
                    raise EditError(
                        "BEVEL_UNSUPPORTED_VERTEX",
                        f"the corner at vertex {v} would be a degenerate polygon",
                        elements={"vertices": [v]},
                        hint="bevel fewer edges at this vertex",
                    )
                chains.append((v, chain))
                continue
            simple = poly.is_simple(poly.project(pts, poly.newell_normal(pts)))
            if simple and (n == 1 or _plane_deviation(pts) <= 1e-9 * scale):
                fills.append((v, hole, owner, "polygon"))
            elif (
                n >= 2
                and closed_at[v]
                and all(e in bevelled for e in self._vert_edges[v])
                and self._patch_sides(hole, inner_points, n) is not None
            ):
                fills.append((v, hole, owner, "patch"))
            else:
                fills.append((v, hole, owner, "fan"))

        for f, x, y, inner in merges:
            loop, corners = new_loops[f]
            k = next(i for i in range(len(loop)) if loop[i] == x and loop[(i + 1) % len(loop)] == y)
            loop[k + 1 : k + 1] = inner
            corners[k + 1 : k + 1] = [None] * len(inner)

        # 6. Strip plans (loops and attributes are completed below).
        strip_plans: list[tuple[int, list[int], list[tuple], bool, int]] = []
        strip_ends: dict[int, tuple] = {}
        strip_sides: dict[int, tuple] = {}
        for e in eids:
            f, g = edge_faces[e]
            a, b = directed[(e, f)]
            xf_a, xg_a = self._point_in(a, f, replacement), self._point_in(a, g, replacement, False)
            xf_b, xg_b = self._point_in(b, f, replacement, False), self._point_in(b, g, replacement)
            pa, pb = profile_path(a, xf_a, xg_a), profile_path(b, xf_b, xg_b)
            strip_ends[e] = (self._pos[b] - self._pos[a], xf_a, xf_b, xg_a, xg_b)
            strip_sides[e] = (f, g, xf_a, xg_a, xf_b, xg_b)
            smooth = (self._smooth[f] and self._smooth[g]) if n == 1 else True
            group = self._group.get(min(f, g), 0)
            for k in range(n):
                loop = [pb[k], pa[k], pa[k + 1], pb[k + 1]]
                params = [("b", k / n), ("a", k / n), ("a", (k + 1) / n), ("b", (k + 1) / n)]
                strip_plans.append((e, loop, params, smooth, group))

        # Flat corners: subdivide both sides of the line by each other's points.
        if chains:
            self._apply_chains(chains, new_loops, strip_plans)

        # 7. Attributes of rebuilt faces (affine fit over the original face), then build.
        corner_attrs: dict[tuple[int, int], dict] = {}
        for f in affected:
            new_loop, new_corners = new_loops[f]
            targets = [
                (i, x)
                for i, (x, c) in enumerate(zip(new_loop, new_corners, strict=True))
                if c is None
            ]
            if targets:
                values = self._affine_attrs(f, np.array([self._pos[x] for _, x in targets]))
                for (slot, _), attrs in zip(targets, values, strict=True):
                    new_corners[slot] = attrs
            for x, attrs in zip(new_loop, new_corners, strict=True):
                corner_attrs[(f, x)] = attrs
            if len(set(new_loop)) != len(new_loop) or len(new_loop) < 3:
                raise EditError(
                    "BEVEL_UNSUPPORTED_TOPOLOGY",
                    f"face {f} would become invalid",
                    elements={"faces": [f]},
                )
        for f, (new_loop, new_corners) in new_loops.items():
            self._replace_face(f, new_loop, self._split_corners(new_corners))

        strips: list[int] = []
        for e, loop, params, smooth, group in strip_plans:
            f, g, xf_a, xg_a, xf_b, xg_b = strip_sides[e]
            ends = {
                "a": (corner_attrs[(f, xf_a)], corner_attrs[(g, xg_a)]),
                "b": (corner_attrs[(f, xf_b)], corner_attrs[(g, xg_b)]),
            }
            attrs = []
            for side, t in params:
                lo_attrs, hi_attrs = ends[side]
                attrs.append(self._mix([(lo_attrs, 1 - t), (hi_attrs, t)]))
            if len(set(loop)) != len(loop):
                raise EditError(
                    "BEVEL_UNSUPPORTED_TOPOLOGY",
                    f"the strip of edge {e} would repeat a vertex",
                    elements={"edges": [e]},
                )
            strips.append(
                self._new_face(loop, self._split_corners(attrs), smooth=smooth, group=group)
            )

        # 8. Corner polygons, patches and fans.
        first_attrs: dict[int, dict] = {}
        for (_f, x), attrs in sorted(corner_attrs.items(), key=lambda item: item[0]):
            first_attrs.setdefault(x, attrs)
        profile_at: dict[tuple[int, int], list[int]] = {}
        for (pv, _lo, _hi), ids in profiles.items():
            for x in ids[1:-1]:
                profile_at[(pv, x)] = ids
        corner_faces: list[int] = []
        for v, hole, owner, kind in fills:
            attrs = [
                self._outline_attrs(x, owner.get(x, -1), v, profile_at, corner_attrs, first_attrs)
                for x in hole
            ]
            smooth = n > 1 or all(self._smooth[f] for f, *_ in fan_of[v])
            group = self._group.get(min(f for f, *_ in fan_of[v]), 0)
            if kind == "polygon":
                corner_faces.append(
                    self._new_face(hole, self._split_corners(attrs), smooth=smooth, group=group)
                )
            elif kind == "patch":
                corner_faces += self._grid_patch(
                    v, hole, attrs, inner_points, n, group, new_vertices
                )
            else:
                corner_faces += self._fan_patch(v, hole, attrs, smooth, group, new_vertices)

        for v in vertices:
            if v not in retained and self._alive[v] and not self._vert_edges[v]:
                self._delete_vertex(v)

        # 9. Verify: strips keep their direction, nothing folded, nothing degenerate.
        for e, (direction, xf_a, xf_b, xg_a, xg_b) in strip_ends.items():
            if (
                float(np.dot(self._pos[xf_b] - self._pos[xf_a], direction)) <= 0
                or float(np.dot(self._pos[xg_b] - self._pos[xg_a], direction)) <= 0
            ):
                raise EditError(
                    "BEVEL_WIDTH_TOO_LARGE",
                    f"width {w} collapses the faces next to edge {e}",
                    elements={"edges": [e]},
                    hint="use a smaller width",
                )
        eps = self._area_eps()
        for f in affected:
            nf = self._face_normal_vec(f)
            if (
                nf is None
                or float(np.dot(nf, original_normals[f])) <= 0
                or not poly.is_simple(poly.project(self._pos[self._faces[f]], original_normals[f]))
            ):
                raise EditError(
                    "BEVEL_WIDTH_TOO_LARGE",
                    f"width {w} folds face {f}",
                    elements={"faces": [f]},
                    hint="use a smaller width",
                )
        bad = [
            f
            for f in [*strips, *corner_faces]
            if poly.polygon_area(self._pos[self._faces[f]]) <= eps
        ]
        if bad:
            raise EditError(
                "BEVEL_WIDTH_TOO_LARGE",
                "the bevel would create zero-area faces",
                elements={"faces": bad},
                hint="use a smaller width or fewer segments",
            )
        return {
            "faces": Selection.of(faces=strips),
            "corner_faces": Selection.of(faces=corner_faces),
            "vertices": Selection.of(v for v in new_vertices if self._alive[v]),
        }

    # -- profiles -----------------------------------------------------------------------

    def _profile_middle(
        self, v: int, e: int, start: int, end: int, mitre_points: set[int]
    ) -> np.ndarray:
        """Where the edge's line crosses the profile's plane (see the module docs)."""
        pv = self._pos[v]
        if start not in mitre_points or end not in mitre_points:
            return pv.copy()
        a, b = self._edge_verts[e]
        d = poly.unit(self._pos[b] - self._pos[a])
        chord = poly.unit(self._pos[end] - self._pos[start])
        if d is None or chord is None:
            return pv.copy()
        nrm = d - float(np.dot(d, chord)) * chord  # the plane's normal: d without the chord
        length = float(np.linalg.norm(nrm))
        if length < 1e-6:
            return pv.copy()
        nrm = nrm / length
        mid = (self._pos[start] + self._pos[end]) / 2.0
        t = float(np.dot(mid - pv, nrm)) / float(np.dot(d, nrm))
        return pv + t * d

    # -- corners ------------------------------------------------------------------------

    def _merge_into_face(
        self, hole: list[int], new_loops: dict, inner_points: set[int]
    ) -> tuple[int, int, int, list[int]] | None:
        """A corner that is one profile plus its chord, next to a face running along the
        chord: that face takes the profile. Returns ``(face, x, y, points)`` to insert
        between ``x`` and ``y``."""
        ends = [x for x in hole if x not in inner_points]
        if len(ends) != 2:
            return None
        i = hole.index(ends[0])
        rot = hole[i:] + hole[:i]
        if rot[1] in inner_points:  # end0 -> inner -> end1 -> (chord) -> end0
            x, inner, y = rot[0], rot[1:-1], rot[-1]
        else:  # end0 -> (chord) -> end1 -> inner -> end0
            x, inner, y = rot[1], rot[2:], rot[0]
        if not inner or any(p not in inner_points for p in inner):
            return None
        # The corner runs x -> inner -> y and back along the chord y -> x; the face on the
        # other side of the chord runs x -> y, and takes the corner's path instead.
        for f, (loop, _corners) in new_loops.items():
            m = len(loop)
            for k in range(m):
                if loop[k] == x and loop[(k + 1) % m] == y:
                    return f, x, y, list(inner)
        return None

    def _collinear_chain(self, hole: list[int]) -> list[int] | None:
        """For a zero-area corner whose points lie on one line: its points ordered along
        the line (or ``None`` if they do not lie on one line)."""
        pts = self._pos[hole]
        span = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(span))
        i0, i1 = int(np.argmin(pts[:, axis])), int(np.argmax(pts[:, axis]))
        d = pts[i1] - pts[i0]
        length = float(np.linalg.norm(d))
        if length == 0.0:
            return None
        u = d / length
        rel = pts - pts[i0]
        off = rel - np.outer(rel @ u, u)
        if float(np.sqrt((off**2).sum(axis=1)).max()) > 1e-9 * max(self._scale(), length):
            return None
        order = np.argsort(rel @ u, kind="stable")
        return [hole[i] for i in order]

    def _apply_chains(self, chains: list, new_loops: dict, strip_plans: list) -> None:
        """Subdivide every edge along a flat corner's line by the line's other points."""
        for _v, chain in chains:
            rank = {x: i for i, x in enumerate(chain)}
            loops: list[tuple[list, list | None]] = [(lp, cs) for lp, cs in new_loops.values()]
            loops += [(plan[1], plan[2]) for plan in strip_plans]
            for loop, extra in loops:
                k = 0
                while k < len(loop):
                    x, y = loop[k], loop[(k + 1) % len(loop)]
                    if x in rank and y in rank and abs(rank[x] - rank[y]) > 1:
                        lo, hi = sorted((rank[x], rank[y]))
                        between = chain[lo + 1 : hi]
                        if rank[x] > rank[y]:
                            between = between[::-1]
                        loop[k + 1 : k + 1] = between
                        if extra is not None:
                            if extra and isinstance(extra[0], tuple):  # strip parameters
                                t0 = extra[k][1]
                                t1 = extra[(k + 1) % len(extra)][1]
                                side = extra[k][0]
                                filler = [
                                    (side, t0 + (t1 - t0) * (j + 1) / (len(between) + 1))
                                    for j in range(len(between))
                                ]
                                extra[k + 1 : k + 1] = filler
                            else:
                                extra[k + 1 : k + 1] = [None] * len(between)
                        k += len(between)
                    k += 1

    def _patch_sides(self, hole: list[int], inner_points: set[int], n: int) -> int | None:
        """The number of sides if the outline is profiles joined at their ends (``n - 1``
        inner points between consecutive ends), else ``None``."""
        corners = [i for i, x in enumerate(hole) if x not in inner_points]
        k = len(corners)
        if k < 3 or len(hole) != k * n:
            return None
        if any((corners[(i + 1) % k] - corners[i]) % len(hole) != n for i in range(k)):
            return None
        return k

    def _outline_attrs(
        self,
        x: int,
        owner: int,
        v: int,
        profile_at: dict,
        corner_attrs: dict,
        first_attrs: dict,
    ) -> dict:
        """Attributes of an outline point: its face's, or interpolated along its profile
        between the faces at the profile's ends (a point several faces share takes the
        lowest-id face's values)."""
        if owner >= 0 and (owner, x) in corner_attrs:
            return corner_attrs[(owner, x)]
        ids = profile_at.get((v, x))
        if ids is not None:
            t = ids.index(x) / (len(ids) - 1)
            return self._mix([(first_attrs[ids[0]], 1 - t), (first_attrs[ids[-1]], t)])
        return first_attrs[x]

    def _fan_patch(
        self, v: int, hole: list[int], attrs: list[dict], smooth: bool, group: int, created: list
    ) -> list[int]:
        """Close a corner with triangles around a new center vertex."""
        pts = self._pos[hole]
        nrm = poly.unit(poly.newell_normal(pts))
        c = self._new_vertex(pts.mean(axis=0))
        created.append(c)
        mid = self._mix([(a, 1.0 / len(attrs)) for a in attrs])
        faces = []
        m = len(hole)
        for i in range(m):
            j = (i + 1) % m
            tri_n = poly.unit(poly.newell_normal(self._pos[[c, hole[i], hole[j]]]))
            if tri_n is None or (nrm is not None and float(np.dot(tri_n, nrm)) <= 0):
                raise EditError(
                    "BEVEL_UNSUPPORTED_VERTEX",
                    f"the corner at vertex {v} cannot be filled without folding",
                    elements={"vertices": [v]},
                    hint="bevel fewer edges at this vertex or use another width",
                )
            faces.append(
                self._new_face(
                    [c, hole[i], hole[j]],
                    self._split_corners([mid, attrs[i], attrs[j]]),
                    smooth=smooth,
                    group=group,
                )
            )
        return faces

    def _grid_patch(
        self,
        v: int,
        hole: list[int],
        attrs: list[dict],
        inner_points: set[int],
        n: int,
        group: int,
        created: list,
    ) -> list[int]:
        """Fill a corner bounded by ``k`` profiles with rings of quads (see module docs)."""
        corners = [i for i, x in enumerate(hole) if x not in inner_points]
        shift = corners[0]
        hole = hole[shift:] + hole[:shift]
        attrs = attrs[shift:] + attrs[:shift]
        k = len(hole) // n
        boundary = self._pos[hole]
        angles = 2.0 * math.pi * np.arange(k) / k
        poly_corners = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        side_t = np.arange(n) / n
        domain = np.concatenate(
            [
                poly_corners[i] + side_t[:, None] * (poly_corners[(i + 1) % k] - poly_corners[i])
                for i in range(k)
            ]
        )
        # Rings j = 1 .. n // 2: k sides of n - 2j segments, scaled toward the center.
        ring_points: list[np.ndarray] = []
        ring_scales: list[float] = []
        for j in range(1, n // 2 + 1):
            s_j = n - 2 * j
            scale_j = 1.0 - 2.0 * j / n
            if s_j == 0:
                ring_points.append(np.zeros((1, 2)))
            else:
                t = np.arange(s_j) / s_j
                ring_points.append(
                    np.concatenate(
                        [
                            scale_j
                            * (
                                poly_corners[i]
                                + t[:, None] * (poly_corners[(i + 1) % k] - poly_corners[i])
                            )
                            for i in range(k)
                        ]
                    )
                )
            ring_scales.append(scale_j)
        interior = np.concatenate(ring_points) if ring_points else np.zeros((0, 2))
        weights = _mean_value_weights(domain, interior) if len(interior) else np.zeros((0, k * n))
        base = weights @ boundary
        # Bulge: the profiles bulge away from their chords; the center bulges by their mean.
        bulges = []
        for i in range(k):
            ids = hole[i * n : (i + 1) * n] + [hole[((i + 1) * n) % len(hole)]]
            pts = self._pos[ids]
            bulges.append(pts[n // 2] if n % 2 == 0 else (pts[n // 2] + pts[n // 2 + 1]) / 2)
            bulges[-1] = bulges[-1] - (pts[0] + pts[-1]) / 2
        bulge = np.mean(bulges, axis=0)
        scales = (
            np.concatenate(
                [np.full(len(rp), sc) for rp, sc in zip(ring_points, ring_scales, strict=True)]
            )
            if ring_points
            else np.zeros(0)
        )
        positions = base + ((1.0 - scales**2)[:, None] * bulge if len(base) else 0.0)
        fit = _fit_sphere(boundary)
        if fit is not None and len(positions):
            sphere_center, radius, dev = fit
            if dev <= 1e-6 * radius:
                rel = positions - sphere_center
                lengths = np.sqrt((rel**2).sum(axis=1))
                ok = lengths > 1e-12 * radius
                positions[ok] = sphere_center + rel[ok] * (radius / lengths[ok])[:, None]
        # Vertices of the rings (ids), ring 0 being the outline itself.
        rings: list[list[int]] = [list(hole)]
        ring_attrs: list[list[dict]] = [list(attrs)]
        offset = 0
        for rp in ring_points:
            ids, ring_a = [], []
            for row in range(len(rp)):
                x = self._new_vertex(positions[offset + row])
                created.append(x)
                ids.append(x)
                wts = weights[offset + row]
                ring_a.append(self._mix([(a, float(wt)) for a, wt in zip(attrs, wts, strict=True)]))
            rings.append(ids)
            ring_attrs.append(ring_a)
            offset += len(rp)
        faces: list[int] = []

        def add(loop: list[tuple[int, int, int]]) -> None:
            vids = [rings[j][i] for j, i, _ in loop]
            vals = [ring_attrs[j][i] for j, i, _ in loop]
            faces.append(self._new_face(vids, self._split_corners(vals), smooth=True, group=group))

        for j in range(len(rings) - 1):
            s_out = n - 2 * j
            s_in = s_out - 2
            out_len = len(rings[j])
            for i in range(k):
                base_o = i * s_out
                if s_in <= 0:
                    # Corner quads around the single center point (even n).
                    add(
                        [
                            (j, (base_o + s_out - 1) % out_len, 0),
                            (j, (base_o + s_out) % out_len, 0),
                            (j, (base_o + s_out + 1) % out_len, 0),
                            (j + 1, 0, 0),
                        ]
                    )
                    continue
                in_len = len(rings[j + 1])
                base_i = i * s_in
                for t in range(1, s_out - 1):
                    add(
                        [
                            (j, (base_o + t) % out_len, 0),
                            (j, (base_o + t + 1) % out_len, 0),
                            (j + 1, (base_i + t) % in_len, 0),
                            (j + 1, (base_i + t - 1) % in_len, 0),
                        ]
                    )
                add(
                    [
                        (j, (base_o + s_out - 1) % out_len, 0),
                        (j, (base_o + s_out) % out_len, 0),
                        (j, (base_o + s_out + 1) % out_len, 0),
                        (j + 1, (base_i + s_in) % in_len, 0),
                    ]
                )
        if n % 2 == 1:
            add([(len(rings) - 1, i, 0) for i in range(len(rings[-1]))])
        # Every patch face must face the same way as the outline.
        nrm = poly.unit(poly.newell_normal(boundary))
        for f in faces:
            fn = self._face_normal_vec(f)
            if fn is None or (nrm is not None and float(np.dot(fn, nrm)) <= 0):
                raise EditError(
                    "BEVEL_UNSUPPORTED_VERTEX",
                    f"the corner patch at vertex {v} would fold",
                    elements={"vertices": [v]},
                    hint="use a smaller width or fewer segments",
                )
        return faces
