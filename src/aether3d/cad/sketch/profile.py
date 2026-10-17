"""Profiles: closed regions bounded by a sketch's curves.

The non-construction curves of a solved sketch are intersected pairwise and split at the
intersections, endpoints closer than the tolerance policy's ``resolution`` are merged, and
dangling chains (open ends) are pruned and reported. The remaining planar graph is traced
with half-edges sorted around each vertex by tangent direction, then by curvature (so a
line tangent to an arc is ordered correctly), giving every face of the arrangement as a
cycle of oriented curve pieces with an exact signed area (Green's theorem on lines, arcs
and splines).

Two ways to choose material:

* ``"auto"`` (the default) -- connected groups of curves are nested by containment; a
  group at even depth (0, 2, ...) is solid, the groups directly inside it are its holes.
  Curves inside a group (a line across a rectangle) only split faces and do not change
  the region: what a closed outline means in every CAD system.
* points -- each given point selects the arrangement face containing it, with the groups
  nested directly inside that face as holes (to extrude one cell of a subdivided outline).

Every piece remembers the sketch entity it came from (for face names) and its parameter
range on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...errors import CadError
from ..geom2d import Arc, Segment, Spline, intersect
from ..nurbs import BSplineCurve
from ..tolerance import Tolerance

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class Piece:
    """A curve piece of a profile loop, oriented along the loop."""

    curve: Segment | Arc | Spline
    source: str  # sketch entity id
    index: int  # piece number on that entity (0 when unsplit)
    split: bool = False  # one of the two halves of a closed curve

    @property
    def start(self) -> np.ndarray:
        return self.curve.start

    @property
    def end(self) -> np.ndarray:
        return self.curve.end


@dataclass
class Loop:
    pieces: list[Piece]

    @property
    def area(self) -> float:
        """Signed area (positive counter-clockwise)."""
        return float(sum(p.curve.signed_area_term() for p in self.pieces))

    def reversed(self) -> Loop:
        return Loop(
            [Piece(p.curve.reversed(), p.source, p.index, p.split) for p in reversed(self.pieces)]
        )

    def polyline(self, chord: float) -> np.ndarray:
        pts = []
        for p in self.pieces:
            ts = p.curve.sample(chord)
            pts.append(p.curve.point(ts[:-1]))
        return np.concatenate(pts, axis=0)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        boxes = [p.curve.bbox() for p in self.pieces]
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)


@dataclass
class Region:
    """A connected planar region: a counter-clockwise outer loop and clockwise holes."""

    outer: Loop
    holes: list[Loop] = field(default_factory=list)

    @property
    def area(self) -> float:
        return self.outer.area + sum(h.area for h in self.holes)

    @property
    def loops(self) -> list[Loop]:
        return [self.outer, *self.holes]


@dataclass
class ProfileResult:
    regions: list[Region]
    open_ends: list[list[float]]
    faces: int
    tolerance: float


def sketch_curves(defn: dict) -> list[tuple[str, Segment | Arc | Spline]]:
    """The non-construction curves of a (solved) definition, in creation order."""
    ents = {e["id"]: e for e in defn.get("entities", [])}
    out = []
    for e in defn.get("entities", []):
        if e.get("construction"):
            continue
        t = e["type"]
        if t == "line":
            out.append((e["id"], Segment(ents[e["start"]]["at"], ents[e["end"]]["at"])))
        elif t == "circle":
            out.append((e["id"], Arc(ents[e["center"]]["at"], float(e["radius"]), 0.0, TAU)))
        elif t == "arc":
            c = np.asarray(ents[e["center"]]["at"], float)
            s = np.asarray(ents[e["start"]]["at"], float)
            f = np.asarray(ents[e["end"]]["at"], float)
            a0 = math.atan2(s[1] - c[1], s[0] - c[0])
            a1 = math.atan2(f[1] - c[1], f[0] - c[0])
            sweep = (a1 - a0) % TAU
            if sweep == 0.0:
                sweep = TAU
            out.append((e["id"], Arc(c, float(e["radius"]), a0, sweep)))
        elif t == "spline":
            pts = np.array([ents[p]["at"] for p in e["points"]], dtype=np.float64)
            deg = int(e.get("degree", 3))
            n = len(pts)
            inner = np.linspace(0.0, 1.0, n - deg + 1)[1:-1]
            knots = np.concatenate([[0.0] * (deg + 1), inner, [1.0] * (deg + 1)])
            out.append((e["id"], Spline(BSplineCurve(deg, knots, pts))))
    return out


def _arc_endpoints_exact(c: Arc, t: float) -> np.ndarray:
    return c.point(t)


def build_profile(
    defn: dict,
    *,
    select: list | None = None,
    tolerance: float | None = None,
    location: str = "sketch",
) -> ProfileResult:
    """Regions of a solved sketch (see the module docstring)."""
    curves = sketch_curves(defn)
    if not curves:
        raise CadError("EMPTY_PROFILE", "the sketch has no (non-construction) curves", location)
    pts_all = np.concatenate(
        [c.bbox()[0][None] for _, c in curves] + [c.bbox()[1][None] for _, c in curves]
    )
    size = float(np.max(np.abs(pts_all))) + float(np.linalg.norm(pts_all.max(0) - pts_all.min(0)))
    tol = tolerance if tolerance is not None else Tolerance.for_scale(size).resolution * 8
    # 1. Split parameters from pairwise intersections (box filter).
    boxes = [c.bbox() for _, c in curves]
    splits: list[list[float]] = [[] for _ in curves]
    order = np.argsort([b[0][0] for b in boxes])
    for ii, i in enumerate(order):
        lo_i, hi_i = boxes[i]
        for j in order[ii + 1 :]:
            lo_j, hi_j = boxes[j]
            if lo_j[0] > hi_i[0] + tol:
                break
            if lo_j[1] > hi_i[1] + tol or lo_i[1] > hi_j[1] + tol:
                continue
            for h in intersect(curves[i][1], curves[j][1], tol):
                splits[i].append(h.ta)
                splits[j].append(h.tb)
                if h.overlap is not None:
                    splits[i].append(h.overlap[0])
                    splits[j].append(h.overlap[1])
    # 2. Pieces.
    pieces: list[tuple[str, int, object, bool]] = []
    for (cid, curve), ts in zip(curves, splits, strict=True):
        length = max(curve.length, 1e-300)
        cuts = sorted(t for t in ts if tol / length < t < 1.0 - tol / length)
        params = [0.0]
        for t in cuts:
            if (t - params[-1]) * length > tol:
                params.append(t)
        if (1.0 - params[-1]) * length <= tol and len(params) > 1:
            params.pop()
        params.append(1.0)
        closed_circle = isinstance(curve, Arc) and curve.closed
        if closed_circle and len(params) == 2:
            # A full circle with no cuts: two halves, so the graph has two vertices.
            params = [0.0, 0.5, 1.0]
        for k, (a, b) in enumerate(zip(params[:-1], params[1:], strict=True)):
            sub = curve.sub(a, b) if (a, b) != (0.0, 1.0) else curve
            pieces.append((cid, k, sub, closed_circle and len(params) == 3))
    # 3. Vertices: merge endpoints within tol.
    ends = np.array([[p[2].start, p[2].end] for p in pieces], dtype=np.float64).reshape(-1, 2)
    vid = _merge_points(ends, tol)
    nv = int(vid.max()) + 1 if len(vid) else 0
    edges = [(int(vid[2 * i]), int(vid[2 * i + 1])) for i in range(len(pieces))]
    # Drop zero-length pieces (both ends merged) that are not closed loops.
    keep = [i for i, (a, b) in enumerate(edges) if a != b or pieces[i][2].length > tol * 4]
    # 4. Prune dangling chains.
    alive = set(keep)
    degree = [0] * nv
    for i in alive:
        a, b = edges[i]
        degree[a] += 1
        degree[b] += 1
    open_ends = [v for v in range(nv) if degree[v] == 1]
    reported = [ends[np.flatnonzero(vid == v)[0]].tolist() for v in open_ends]
    stack = list(open_ends)
    while stack:
        v = stack.pop()
        if degree[v] != 1:
            continue
        for i in list(alive):
            a, b = edges[i]
            if a == v or b == v:
                alive.discard(i)
                degree[a] -= 1
                degree[b] -= 1
                other = b if a == v else a
                if degree[other] == 1:
                    stack.append(other)
                break
    if not alive:
        raise CadError(
            "OPEN_PROFILE",
            "the sketch's curves do not close any region",
            location,
            hint="make the ends meet (coincident constraints) or close the outline",
            details={"open_ends": reported[:20]},
        )
    # 5. Half-edges and face tracing.
    half: list[tuple[int, int, int, bool]] = []  # (piece, from, to, reversed)
    out_at: dict[int, list[int]] = {}
    for i in sorted(alive):
        a, b = edges[i]
        for rev in (False, True):
            h = len(half)
            half.append((i, b if rev else a, a if rev else b, rev))
            out_at.setdefault(half[h][1], []).append(h)
    curve_of = {}

    def oriented(h: int):
        if h not in curve_of:
            i, _, _, rev = half[h]
            c = pieces[i][2]
            curve_of[h] = c.reversed() if rev else c
        return curve_of[h]

    def key(h: int) -> tuple[float, float]:
        c = oriented(h)
        d = c.tangent(0.0)
        ang = math.atan2(float(d[1]), float(d[0])) % TAU
        k = float(np.atleast_1d(c.curvature(np.array([0.0])))[0])
        return ang, k

    sorted_out: dict[int, list[int]] = {}
    position: dict[int, tuple[int, int]] = {}
    for v, hs in out_at.items():
        hs = sorted(hs, key=key)
        sorted_out[v] = hs
        for k, h in enumerate(hs):
            position[h] = (v, k)
    twin = {h: h ^ 1 for h in range(len(half))}
    nxt = {}
    for h in range(len(half)):
        t = twin[h]
        v, k = position[t]
        ring = sorted_out[v]
        nxt[h] = ring[(k - 1) % len(ring)]
    seen: set[int] = set()
    cycles: list[list[int]] = []
    for h in range(len(half)):
        if h in seen:
            continue
        cyc = []
        cur = h
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = nxt[cur]
            if len(cyc) > len(half):
                raise CadError("PROFILE_FAILED", "could not trace the sketch's faces", location)
        cycles.append(cyc)

    def loop_of(cyc: list[int]) -> Loop:
        return Loop(
            [Piece(oriented(h), *pieces[half[h][0]][:2], split=pieces[half[h][0]][3]) for h in cyc]
        )

    loops = [loop_of(c) for c in cycles]
    areas = [lp.area for lp in loops]
    # 6. Components of the graph and their outer (negative) cycles.
    comp = _components(nv, [edges[i] for i in sorted(alive)])
    cyc_comp = [comp[half[c[0]][1]] for c in cycles]
    outer_of: dict[int, int] = {}
    for ci, (cc, a) in enumerate(zip(cyc_comp, areas, strict=True)):
        if a < 0 and (cc not in outer_of or a < areas[outer_of[cc]]):
            outer_of[cc] = ci
    chord = max(size * 1e-6, tol * 4)
    polys = {ci: loops[ci].polyline(chord) for ci in range(len(loops))}
    comps = sorted(outer_of)
    sample = {cc: _sample_point(loops[outer_of[cc]]) for cc in comps}
    # Containment of components: A contains B if B's sample is inside A's outer boundary.
    contains = {
        a: [b for b in comps if b != a and _inside(sample[b], polys[outer_of[a]])] for a in comps
    }
    depth = {cc: sum(1 for a in comps if cc in contains[a]) for cc in comps}
    regions: list[Region] = []
    if select is None:
        for cc in comps:
            if depth[cc] % 2:
                continue
            outer = loops[outer_of[cc]].reversed()
            holes = [loops[outer_of[b]] for b in contains[cc] if depth[b] == depth[cc] + 1]
            regions.append(Region(outer, holes))
    else:
        bounded = [ci for ci, a in enumerate(areas) if a > 0]
        for k, pt in enumerate(select):
            p = np.asarray(pt, dtype=np.float64).reshape(2)
            hits = [ci for ci in bounded if _inside(p, polys[ci])]
            if not hits:
                raise CadError(
                    "NO_REGION_AT_POINT",
                    f"no closed region of the sketch contains the point {p.tolist()}",
                    f"{location}.regions[{k}]",
                )
            ci = min(hits, key=lambda i: areas[i])
            cc = cyc_comp[ci]
            holes = [
                loops[outer_of[b]]
                for b in comps
                if b != cc
                and _inside(sample[b], polys[ci])
                and not any(
                    other != cc and b in contains[other] and _inside(sample[other], polys[ci])
                    for other in comps
                    if other != b
                )
            ]
            regions.append(Region(loops[ci], holes))
    return ProfileResult(regions, reported, len(cycles), tol)


def _merge_points(pts: np.ndarray, tol: float) -> np.ndarray:
    """Cluster ids: points within ``tol`` (single linkage on a grid)."""
    n = len(pts)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell = max(tol, 1e-300)
    grid: dict[tuple[int, int], list[int]] = {}
    keys = np.floor(pts / cell).astype(np.int64)
    for i in range(n):
        kx, ky = int(keys[i, 0]), int(keys[i, 1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((kx + dx, ky + dy), ()):
                    if np.hypot(*(pts[i] - pts[j])) <= tol:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[max(ri, rj)] = min(ri, rj)
        grid.setdefault((kx, ky), []).append(i)
    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(r, len(remap)) for r in roots], dtype=np.int64)


def _components(nv: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(nv))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(v) for v in range(nv)]


def _sample_point(loop: Loop) -> np.ndarray:
    return loop.pieces[0].curve.point(0.5)


def _inside(p: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd point in polygon (the polygon approximates a curved loop finely)."""
    x, y = float(p[0]), float(p[1])
    xs, ys = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (yn > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xcross = xs + (y - ys) * (xn - xs) / (yn - ys)
    return bool(np.count_nonzero(cond & (x < xcross)) % 2)
