"""Planar cuts (bisect) and knife cuts projected onto the surface.

Bisect
------
Every vertex of the cut faces is classified once by its signed distance ``s`` to the
plane: on the plane if ``|s| <= threshold``, in front or behind otherwise. Classifying
each vertex once keeps every later decision consistent (an edge is crossed exactly when
its two ends are strictly on opposite sides). Crossed edges are split at
``t = s_a / (s_a - s_b)`` and the new vertex is projected onto the plane; then each cut
face is split by the chord between its two crossings, which run from the end of the
on-plane run before the face's front stretch to the start of the run after it. A face
crossing the plane more than twice (a concave polygon) is triangulated first, and each
triangle is split (reported as ``BISECT_TRIANGULATED``). Faces outside the selection only
gain the cut points on edges they share with cut faces, so no T-junction is created.

Knife
-----
:meth:`CutMixin.knife_project` cuts along a polyline swept along a direction: every
segment ``P_i P_i+1`` spans a planar *ribbon* ``{P_i + s (P_i+1 - P_i) + t d}``,
``0 <= s <= 1``. In each face the ribbons leave segments, which join at the polyline's
corners into paths; a path that enters and leaves the face splits it (several paths
split it into several pieces). Paths ending inside a face are not applied (a polygon
mesh cannot hold a dangling edge) and are reported as ``KNIFE_PARTIAL_CUT``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import EditError, ValidationError
from ..math3d import vec3
from . import _polygon as poly
from .core import Elements, Selection, nested_data, operation


def _plane(point: Iterable[float], normal: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(vec3(point, "point"), dtype=np.float64)
    n = poly.unit(np.asarray(vec3(normal, "normal"), dtype=np.float64))
    if n is None:
        raise ValidationError("normal must be a non-zero vector")
    return o, n


class CutMixin:
    @operation
    def bisect(
        self,
        point: Iterable[float],
        normal: Iterable[float],
        *,
        faces: Elements | None = None,
        clear: str = "none",
        fill: bool = False,
        threshold: float | None = None,
        space: str = "local",
    ):
        """Cut faces with the plane through ``point`` with ``normal`` (see the module docs).

        ``clear="front"`` deletes the cut faces on the normal's side, ``"back"`` those
        behind it (faces lying in the plane are kept). ``fill=True`` closes the boundary
        loops the clearing leaves in the plane with polygons. ``threshold`` (default
        ``1e-6`` times the mesh size) is the distance within which a vertex counts as on
        the plane. ``data``: ``"cut_vertices"`` (vertices in the plane, new and old, of
        the cut faces), ``"cut_edges"``, ``"front"``/``"back"`` (faces on each side) and
        ``"fill_faces"``.
        """
        o, n = _plane(point, normal)
        if space == "world":
            o = self._world_to_local(o[None])[0]
            n = poly.unit(self.world_matrix[:3, :3].T @ n)
            if n is None:
                raise EditError("SINGULAR_WORLD_TRANSFORM", "the object's transform is singular")
        elif space != "local":
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        if clear not in ("none", "front", "back"):
            raise ValidationError(f"clear must be 'none', 'front' or 'back', got {clear!r}")
        if fill and clear == "none":
            raise ValidationError("fill needs clear='front' or 'back' (it closes the cut)")
        fids = self.face_ids() if faces is None else self._elements(faces, "faces")
        if threshold is None:
            eps = 1e-6 * self._scale()
        else:
            eps = float(threshold)
            if not (eps >= 0 and np.isfinite(eps)):
                raise ValidationError("threshold must be a finite number >= 0")
        verts = sorted({v for f in fids for v in self._faces[f]})
        dist = dict(zip(verts, ((self._pos[verts] - o) @ n).tolist(), strict=True))

        def side(v: int) -> int:
            s = dist[v]
            return 0 if abs(s) <= eps else (1 if s > 0 else -1)

        # 1. Split the edges whose ends are strictly on opposite sides.
        crossed = sorted(
            {
                e
                for f in fids
                for e in self.face_edges(f)
                if side(self._edge_verts[e][0]) * side(self._edge_verts[e][1]) < 0
            }
        )
        cut_new: list[int] = []
        for e in crossed:
            a, b = self._edge_verts[e]
            t = dist[a] / (dist[a] - dist[b])
            (x,) = self._split_edge(e, a, [t])
            p = self._pos[x] - ((self._pos[x] - o) @ n) * n
            self._set_positions([x], p[None], dirty=False)
            dist[x] = 0.0
            cut_new.append(x)
        # 2. Split each cut face by its chord (triangulating faces crossed more than twice).
        triangulated: list[int] = []
        work = list(fids)
        pieces: set[int] = set(fids)
        while work:
            f = work.pop(0)
            loop = self._faces[f]
            sides = [side(v) for v in loop]
            if min(sides) >= 0 or max(sides) <= 0:
                continue
            chord = _chord(loop, sides)
            if chord is None:
                tris = nested_data(self.triangulate_faces([f]))["faces"][f]
                triangulated.append(f)
                pieces.update(tris)
                work.extend(tris)
                continue
            a, b = chord
            pieces.add(nested_data(self.split_face(f, a, b))["face"])
        if triangulated:
            self._diagnose(
                Diagnostic(
                    "BISECT_TRIANGULATED",
                    INFO,
                    "faces crossing the plane more than twice were triangulated before the cut",
                    None,
                    len(triangulated),
                    tuple(triangulated[:10]),
                )
            )
        pieces &= set(self._faces)
        front = sorted(f for f in pieces if max(side(v) for v in self._faces[f]) > 0)
        back = sorted(f for f in pieces if min(side(v) for v in self._faces[f]) < 0)
        on_plane = sorted({v for f in pieces for v in self._faces[f] if side(v) == 0})
        cut_edges = sorted(
            {
                e
                for f in pieces
                for e in self.face_edges(f)
                if side(self._edge_verts[e][0]) == 0 and side(self._edge_verts[e][1]) == 0
            }
        )
        fill_faces: list[int] = []
        if clear != "none":
            doomed = front if clear == "front" else back
            self._delete_faces_and_loose(doomed, keep_loose=False)
            if fill:
                loops = [
                    edges
                    for edges in self._boundary_edge_loops(
                        [e for e in cut_edges if e in self._edge_verts]
                    )
                ]
                for edges in loops:
                    fill_faces += self._fill_loop(edges, "auto")
            kept = back if clear == "front" else front
            front, back = ([], kept) if clear == "front" else (kept, [])
        return {
            "cut_vertices": Selection.of(v for v in on_plane if self.has_vertex(v)),
            "cut_edges": Selection.of(edges=[e for e in cut_edges if e in self._edge_verts]),
            "front": Selection.of(faces=[f for f in front if f in self._faces]),
            "back": Selection.of(faces=[f for f in back if f in self._faces]),
            "fill_faces": Selection.of(faces=fill_faces),
        }


def _chord(loop: list[int], sides: list[int]) -> tuple[int, int] | None:
    """The two vertices joined by the cut through a face crossing the plane twice, or
    ``None`` if it crosses more often (or touches the plane between its crossings)."""
    m = len(loop)
    # Maximal runs of on-plane vertices with the sides before and after them.
    start = next(i for i in range(m) if sides[i] != 0)
    runs: list[tuple[int, int, int, int]] = []  # (first, last, side before, side after)
    i = (start + 1) % m
    prev_side = sides[start]
    steps = 0
    while steps < m:
        if sides[i] == 0:
            first = i
            while sides[i] == 0:
                i = (i + 1) % m
                steps += 1
            last = (i - 1) % m
            runs.append((first, last, prev_side, sides[i]))
        prev_side = sides[i]
        i = (i + 1) % m
        steps += 1
    crossings = [r for r in runs if r[2] != r[3]]
    if len(crossings) != 2:
        return None
    # The chord runs from the end of the run entering the front stretch to the start of
    # the run leaving it (the other part keeps both runs, which lie along the cut line).
    enter = next(r for r in crossings if r[3] > 0)
    leave = next(r for r in crossings if r[2] > 0)
    a, b = loop[enter[1]], loop[leave[0]]
    ia, ib = loop.index(a), loop.index(b)
    if (ib - ia) % m in (1, m - 1):
        return None  # already joined by an edge: the cut runs along it
    return a, b


class KnifeMixin:
    @operation
    def knife_project(
        self,
        points: Iterable[Iterable[float]],
        direction: Iterable[float],
        *,
        faces: Elements | None = None,
        closed: bool = False,
        through: bool = True,
        space: str = "local",
    ):
        """Cut faces along the polyline ``points`` swept along ``direction`` (see the
        module docs). ``through=False`` cuts only faces whose front faces the cut (normal
        against ``direction``). ``closed`` joins the last point to the first.

        ``data``: ``"vertices"`` (new cut vertices), ``"edges"`` (edges along the cut),
        ``"faces"`` (the pieces of every split face).
        """
        pts = [np.asarray(vec3(p, "points"), dtype=np.float64) for p in points]
        if len(pts) < 2:
            raise ValidationError("a knife cut needs at least 2 points")
        if len(pts) > 10_000:
            raise ValidationError("a knife cut takes at most 10,000 points")
        d = poly.unit(np.asarray(vec3(direction, "direction"), dtype=np.float64))
        if d is None:
            raise ValidationError("direction must be non-zero")
        if space == "world":
            pts = list(self._world_to_local(np.array(pts)))
            d = poly.unit(self._world_direction_to_local(d))
            if d is None:
                raise EditError("SINGULAR_WORLD_TRANSFORM", "the object's transform is singular")
        elif space != "local":
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        if closed:
            pts.append(pts[0])
        ribbons = []
        for i in range(len(pts) - 1):
            seg = pts[i + 1] - pts[i]
            lateral = seg - float(np.dot(seg, d)) * d
            nrm = poly.unit(poly.cross3(seg, d))
            if nrm is None or float(np.dot(lateral, lateral)) == 0.0:
                raise ValidationError(f"segment {i} of the cut runs along the direction")
            ribbons.append((pts[i], nrm, lateral, float(np.dot(lateral, lateral))))
        fids = self.face_ids() if faces is None else self._elements(faces, "faces")
        if not through:
            fids = [
                f
                for f in fids
                if (n := self._face_normal_vec(f)) is not None and float(np.dot(n, d)) < 0
            ]
        eps = 1e-9 * self._scale()
        # Faces the polyline's projection meets: candidates by a box test per ribbon.
        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
        candidates = []
        for f in fids:
            fp = self._pos[self._faces[f]]
            # Project the face onto the plane across the direction and test overlap
            # with the polyline's projected box (a cheap, conservative filter).
            flo = (fp - np.outer(fp @ d, d)).min(axis=0)
            fhi = (fp - np.outer(fp @ d, d)).max(axis=0)
            plo = (np.array(pts) - np.outer(np.array(pts) @ d, d)).min(axis=0)
            phi = (np.array(pts) - np.outer(np.array(pts) @ d, d)).max(axis=0)
            if np.all(flo <= phi + eps) and np.all(fhi >= plo - eps):
                candidates.append(f)
        del lo, hi
        # Concave faces would need several chords per ribbon: triangulate the cut ones.
        triangulated = []
        work = []
        for f in candidates:
            loop = self._faces[f]
            fp = self._pos[loop]
            nf = poly.newell_normal(fp)
            if poly.unit(nf) is None:
                continue
            convex = _is_convex(poly.project(fp, nf))
            crossed = any(_straddles(fp, r, eps) for r in ribbons)
            if crossed and not convex and len(loop) > 3:
                pieces = nested_data(self.triangulate_faces([f]))["faces"][f]
                triangulated.append(f)
                work.extend(pieces)
            elif crossed:
                work.append(f)
        if triangulated:
            self._diagnose(
                Diagnostic(
                    "KNIFE_TRIANGULATED",
                    INFO,
                    "concave faces crossed by the cut were triangulated first",
                    None,
                    len(triangulated),
                    tuple(triangulated[:10]),
                )
            )
        # 1. Pieces of the cut in each face, as (start, end) with endpoints either on the
        # boundary ("e", edge, t, ribbon) / ("v", vertex) or inside ("p", polyline index).
        face_pieces: dict[int, list] = {}
        for f in work:
            loop = self._faces[f]
            fp = self._pos[loop]
            for ri, (origin, nrm, lateral, ll) in enumerate(ribbons):
                piece = _ribbon_piece(self, f, loop, fp, ri, origin, nrm, lateral, ll, d, eps)
                if piece is not None:
                    face_pieces.setdefault(f, []).append(piece)
        # 2. Join pieces into paths through the polyline's corners.
        paths: dict[int, list[list]] = {}
        partial = []
        for f, pieces in face_pieces.items():
            joined = _join_pieces(pieces)
            for path in joined:
                if path[0][0] == "p" or path[-1][0] == "p":
                    partial.append(f)
                    continue
                paths.setdefault(f, []).append(path)
        if partial:
            self._diagnose(
                Diagnostic(
                    "KNIFE_PARTIAL_CUT",
                    INFO,
                    "cuts that end inside a face were not applied (a face cannot hold a "
                    "dangling edge)",
                    None,
                    len(set(partial)),
                    tuple(sorted(set(partial))[:10]),
                )
            )
        # 3. Vertices on edges: one per (edge, ribbon) crossing, in order along each edge.
        on_edge: dict[int, dict[int, float]] = {}
        for fpaths in paths.values():
            for path in fpaths:
                for node in (path[0], path[-1]):
                    if node[0] == "e":
                        on_edge.setdefault(node[1], {})[node[3]] = node[2]
        created: list[int] = []
        edge_vertex: dict[tuple[int, int], int] = {}
        for e in sorted(on_edge):
            a, b = self._edge_verts[e]
            ts = sorted(on_edge[e].items(), key=lambda item: item[1])
            new = self._split_edge(e, a, [t for _, t in ts])
            created += new
            for (ri, _t), x in zip(ts, new, strict=True):
                edge_vertex[(e, ri)] = x
        # 4. Split each face along its paths, one after the other.
        pieces_out: set[int] = set()
        cut_edges: list[int] = []
        for f in sorted(paths):
            parts = {f}
            fits = {}
            for path in paths[f]:
                ends = []
                for node in (path[0], path[-1]):
                    if node[0] == "e":
                        ends.append(edge_vertex[(node[1], node[3])])
                    else:
                        ends.append(node[1])
                inner = [n for n in path[1:-1]]
                inner_ids = []
                for node in inner:
                    key = (f, node[1])
                    if key not in fits:
                        fits[key] = self._new_vertex(node[2])
                        created.append(fits[key])
                    inner_ids.append(fits[key])
                if ends[0] == ends[1]:
                    continue
                host = self._face_holding(parts, ends[0], ends[1])
                if host is None:
                    raise EditError(
                        "KNIFE_INVALID_CUT",
                        f"the cut paths in face {f} cross each other",
                        elements={"faces": [f]},
                    )
                g, new_face, new_edges = self._split_face_path(host, ends[0], ends[1], inner_ids)
                parts.update((g, new_face))
                cut_edges += new_edges
            pieces_out |= parts
        return {
            "vertices": Selection.of(v for v in created if self.has_vertex(v)),
            "edges": Selection.of(edges=[e for e in cut_edges if e in self._edge_verts]),
            "faces": Selection.of(faces=[f for f in pieces_out if f in self._faces]),
        }

    def _face_holding(self, parts: set[int], a: int, b: int) -> int | None:
        """The piece of a split face holding both ``a`` and ``b``."""
        for g in sorted(parts):
            if g in self._faces and a in self._faces[g] and b in self._faces[g]:
                return g
        return None

    def _split_face_path(
        self, g: int, a: int, b: int, inner: list[int]
    ) -> tuple[int, int, list[int]]:
        loop = self._faces[g]
        corners = self._corners_of(g)
        i, j = loop.index(a), loop.index(b)
        m = len(loop)
        if not inner and (j - i) % m in (1, m - 1):
            return g, g, [self.edge_between(a, b)]
        first = [(i + k) % m for k in range((j - i) % m + 1)]  # a .. b along the loop
        second = [(j + k) % m for k in range((i - j) % m + 1)]  # b .. a
        inner_attrs = self._affine_attrs(g, self._pos[inner]) if inner else []
        loop1 = [loop[k] for k in first] + inner[::-1]
        corners1 = [corners[k] for k in first] + inner_attrs[::-1]
        loop2 = [loop[k] for k in second] + inner
        corners2 = [corners[k] for k in second] + inner_attrs
        nrm = poly.newell_normal(self._pos[loop])
        for part in (loop1, loop2):
            pts = self._pos[part]
            if (
                len(set(part)) != len(part)
                or poly.polygon_area(pts) <= self._area_eps()
                or not poly.is_simple(poly.project(pts, nrm))
            ):
                raise EditError(
                    "KNIFE_INVALID_CUT",
                    f"the cut would make an invalid piece of face {g}",
                    elements={"faces": [g]},
                    hint="move the cut away from the face's corners",
                )
        self._replace_face(g, loop1, self._split_corners(corners1))
        new = self._new_face(
            loop2,
            self._split_corners(corners2),
            smooth=self._smooth[g],
            group=self._group.get(g, 0),
        )
        path = [a, *inner, b]
        edges = [self.edge_between(x, y) for x, y in zip(path, path[1:], strict=False)]
        return g, new, edges


def _is_convex(pts2d: np.ndarray) -> bool:
    n = len(pts2d)
    if n <= 3:
        return True
    sign = 0.0
    for k in range(n):
        a, b, c = pts2d[k], pts2d[(k + 1) % n], pts2d[(k + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross < 0:
            return False
        sign += cross
    return sign > 0


def _straddles(fp: np.ndarray, ribbon, eps: float) -> bool:
    origin, nrm, lateral, ll = ribbon
    dist = (fp - origin) @ nrm
    if not (dist.min() < -eps and dist.max() > eps):
        return False
    s = ((fp - origin) @ lateral) / ll
    return not (s.max() < 0 or s.min() > 1)


def _ribbon_piece(mesh, f, loop, fp, ri, origin, nrm, lateral, ll, d, eps):
    """The segment of ribbon ``ri`` inside convex face ``f``, clipped to the ribbon's
    extent, as ``(start, end)`` nodes; ``None`` if the ribbon misses the face."""
    dist = (fp - origin) @ nrm
    side = np.where(np.abs(dist) <= eps, 0, np.sign(dist)).astype(int)
    if not (side.min() < 0 and side.max() > 0):
        return None
    m = len(loop)
    crossings = []  # (node, point)
    for k in range(m):
        k2 = (k + 1) % m
        if side[k] == 0:
            crossings.append((("v", loop[k]), fp[k]))
        elif side[k] * side[k2] < 0:
            a, b = loop[k], loop[k2]
            e = mesh.edge_between(a, b)
            lo_v = mesh._edge_verts[e][0]
            da, db = float(dist[k]), float(dist[k2])
            if lo_v != a:
                da, db = db, da
            t = da / (da - db)
            pa, pb = mesh._pos[lo_v], mesh._pos[mesh._edge_verts[e][1]]
            crossings.append((("e", e, t, ri), pa + t * (pb - pa)))
    unique = []
    for node, point in crossings:
        if all(node != other for other, _ in unique):
            unique.append((node, point))
    if len(unique) != 2:
        return None
    (n0, p0), (n1, p1) = unique
    s0 = float((p0 - origin) @ lateral) / ll
    s1 = float((p1 - origin) @ lateral) / ll
    if s0 > s1:
        (n0, p0, s0), (n1, p1, s1) = (n1, p1, s1), (n0, p0, s0)
    if s1 < 0 or s0 > 1:
        return None
    face_n = poly.unit(poly.newell_normal(fp))

    def corner(index: int) -> tuple:
        # The polyline's point ``index`` projected along d onto the face's plane.
        q = origin if index == ri else origin + lateral
        denom = float(np.dot(d, face_n))
        t = float(np.dot(fp[0] - q, face_n)) / denom if denom else 0.0
        return ("p", index, q + t * d)

    start = n0 if s0 >= 0 else corner(ri)
    end = n1 if s1 <= 1 else corner(ri + 1)
    return (start, end, ri)


def _join_pieces(pieces: list) -> list[list]:
    """Chain pieces sharing polyline corners into paths of nodes."""
    by_corner: dict[int, list[int]] = {}
    for k, (start, end, _ri) in enumerate(pieces):
        for node in (start, end):
            if node[0] == "p":
                by_corner.setdefault(node[1], []).append(k)
    used: set[int] = set()
    paths = []
    for k in sorted(range(len(pieces)), key=lambda i: pieces[i][2]):
        if k in used:
            continue
        used.add(k)
        start, end, _ = pieces[k]
        path = [start, end]
        # Extend at the end, then at the start, through shared corners.
        for forward in (True, False):
            node = path[-1] if forward else path[0]
            while node[0] == "p":
                nxt = [j for j in by_corner.get(node[1], []) if j not in used]
                if not nxt:
                    break
                j = nxt[0]
                used.add(j)
                s, e, _ = pieces[j]
                other = e if s[0] == "p" and s[1] == node[1] else s
                if forward:
                    path.append(other)
                else:
                    path.insert(0, other)
                node = other
        paths.append(path)
    return paths
