"""Orthographic views of a CAD document with hidden lines, and sections.

**What is drawn.** Every edge of every solid, as its exact curve sampled within the
drawing's chord tolerance, except seams between faces of one surface (the halves of a
cylinder); and the silhouettes of curved faces, traced on their tessellation (within the
same chord). Edges are classified visible or hidden by casting rays from points along
them toward the viewer through the tessellated solids (Aether3D's watertight ray
tracer); a run of points is split where the classification changes, so the boundary
between a visible and a hidden part of an edge is placed within ``step`` (1/1000 of the
view's extent by default).

**Sections.** A section plane (a plan's cut at a height above the level, or any plane)
removes the material on the viewer's side by an exact boolean with a half-space; the
faces the cut creates on the plane are the section (poché): exact loops, hatched, with
their outlines in the heavier section line style. Solids not crossing the plane are kept
or dropped whole.

These are geometric drawings: they do not apply drafting standards' conventions (hidden
lines of threads, broken views, symbolic representation of fasteners).
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

import numpy as np

from ...errors import CadError
from ..brep.boolean import boolean, same_surface
from ..brep.tessellate import tessellate
from ..brep.topology import Body
from ..geom3d import Frame
from ..tolerance import TessellationTolerance
from .model import Drawing

MAX_SOLIDS = 5000
MAX_RAYS = 4_000_000

VIEWS = {
    "top": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    "bottom": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "front": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    "back": ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "iso": ((-1.0, 1.0, -1.0), (1.0, 1.0, 0.0)),
}


@dataclass
class Solid:
    name: str
    body: Body  # world coordinates, metres
    role: str
    item: str


@dataclass(frozen=True)
class ViewFrame:
    direction: np.ndarray  # the viewing direction (from the viewer into the scene)
    right: np.ndarray
    up: np.ndarray

    @classmethod
    def named(cls, name: str) -> ViewFrame:
        if name not in VIEWS:
            raise CadError("INVALID_VALUE", f"unknown view {name!r}", hint=", ".join(VIEWS))
        d, r = VIEWS[name]
        return cls.make(d, r)

    @classmethod
    def make(cls, direction, right) -> ViewFrame:
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        r = np.asarray(right, dtype=np.float64)
        r = r - (r @ d) * d
        if np.linalg.norm(r) < 1e-9:
            raise CadError("INVALID_VALUE", "the view's right direction is along its direction")
        r = r / np.linalg.norm(r)
        return cls(d, r, np.cross(r, d))

    def project(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        return np.c_[pts @ self.right, pts @ self.up]


@dataclass(frozen=True)
class Cut:
    origin: np.ndarray
    normal: np.ndarray  # pointing to the viewer's side (the material removed)


# --------------------------------------------------------------------------- #
# Gathering solids from a document
# --------------------------------------------------------------------------- #


def gather(doc, items=None, *, level: str | None = None) -> list[Solid]:
    """The solids of document items (parts, assemblies, building floors and elements),
    in world coordinates. ``level`` keeps the building items of that level only."""
    chosen = [doc.item(i) for i in items] if items is not None else doc.items()
    out: list[Solid] = []
    in_assembly: set[str] = set()
    for it in chosen:
        if it.kind == "assembly":
            res = doc.result(it)
            for leaf in res.leaves:
                in_assembly.add(leaf.part)
                body = doc.result(leaf.part).bodies[leaf.body].transformed(leaf.placement)
                out.append(Solid(f"{it.name}/{leaf.path}", body, "part", it.name))
    for it in chosen:
        k = it.kind
        if k == "part" and (items is not None or it.id not in in_assembly):
            for name, body in sorted(doc.result(it).bodies.items()):
                out.append(Solid(f"{it.name}/{name}", body, "part", it.name))
        elif k == "floor":
            r = doc.result(it)
            if level is not None and r.level.name != level:
                continue
            for wid, wr in r.walls.items():
                out.append(Solid(f"{it.name}/{wid}", wr.body, "wall", it.name))
            for oid, o in r.openings.items():
                if o.body is not None:
                    out.append(
                        Solid(f"{it.name}/{oid}", o.body.transformed(o.placement), o.kind, it.name)
                    )
        elif k in ("slab", "roof", "stair", "columns", "beams"):
            d = it._definition
            if level is not None and d.get("level", d.get("from")) != level:
                continue
            r = doc.result(it)
            for name, body, m in r.bodies:
                out.append(Solid(f"{it.name}/{name}", body.transformed(m), r.role, it.name))
        if len(out) > MAX_SOLIDS:
            raise CadError("LIMIT_EXCEEDED", f"a drawing shows at most {MAX_SOLIDS} solids")
    return out


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def _halfspace(origin, normal, size) -> Body:
    """The half-space {(p - origin) . normal <= 0} within ``size`` of the origin."""
    from ..brep import build

    n = normal / np.linalg.norm(normal)
    x = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    # build.box is centred on its frame: the top face lies on the plane.
    fr = Frame.from_axes(origin - n * size, n, x / np.linalg.norm(x))
    return build.box((4 * size, 4 * size, 2 * size), frame=fr)


def _corners(body: Body) -> np.ndarray:
    lo, hi = body.bounds
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


def project(
    solids: list[Solid],
    view: ViewFrame,
    *,
    title: str,
    view_name: str,
    cut: Cut | None = None,
    hidden: bool = True,
    chord: float | None = None,
    hatch: dict[str, tuple[str, float]] | None = None,
    deadline: float | None = None,
) -> Drawing:
    import time

    drawing = Drawing(title=title, view=view_name)
    if not solids:
        raise CadError("NOTHING_TO_DRAW", "there is nothing to draw")
    lo = np.min([s.body.bounds[0] for s in solids], axis=0)
    hi = np.max([s.body.bounds[1] for s in solids], axis=0)
    extent = float(np.linalg.norm(hi - lo))
    center = 0.5 * (lo + hi)
    chord = chord or extent * 4e-4
    tol = TessellationTolerance(chord, math.radians(10.0), None)
    # 1. Sections.
    kept: list[tuple[Solid, Body, set[int]]] = []
    for s in solids:
        if deadline is not None and time.monotonic() > deadline:
            raise CadError("TIME_BUDGET_EXCEEDED", "drawing ran out of its time budget")
        if cut is None:
            kept.append((s, s.body, set()))
            continue
        side = (_corners(s.body) - cut.origin) @ cut.normal
        eps = 1e-9 * max(extent, 1.0)
        if np.all(side <= eps):
            kept.append((s, s.body, set()))
            continue
        if np.all(side >= -eps):
            continue
        try:
            body = boolean(s.body, _halfspace(cut.origin, cut.normal, extent + 1.0), "intersection")
        except CadError as exc:
            drawing.notes.append(f"{s.name}: not sectioned ({exc.code}); drawn uncut")
            kept.append((s, s.body, set()))
            continue
        caps = set()
        for fi, f in enumerate(body.faces):
            if f.surface.kind != "plane":
                continue
            fz = f.surface.frame.z if f.sense else -f.surface.frame.z
            off = float((f.surface.frame.origin - cut.origin) @ cut.normal)
            if float(fz @ cut.normal) > 1 - 1e-9 and abs(off) <= 1e-9 * max(extent, 1.0):
                caps.add(fi)
        kept.append((s, body, caps))
    # 2. Tessellations and the ray tracing structure.
    tess = []
    for s, body, _ in kept:
        try:
            tess.append(tessellate(body, tol))
        except CadError as exc:
            raise CadError(exc.code, f"{s.name}: {exc.message}", hint=exc.hint) from None
    accel = _accel(tess, center)
    # 3. Section fills.
    style_of = hatch or {}
    for (s, body, caps), t in zip(kept, tess, strict=True):
        for fi in sorted(caps):
            loops = []
            for loop in body.faces[fi].loops:
                pts = _loop_points(body, t, loop)
                loops.append(view.project(pts))
            pattern, gray = style_of.get(s.role, ("solid", 0.72))
            drawing.hatch(loops, layer=f"section-{s.role}", pattern=pattern, gray=gray)
    # 4. Lines.
    step = extent / 1000.0
    lines = []  # (points (N, 3), on_cut, role)
    for (s, body, caps), t in zip(kept, tess, strict=True):
        faces_of = body.edge_faces()
        for ei in range(len(body.edges)):
            adj = [fi for fi, _ in faces_of.get(ei, [])]
            if len(adj) == 2 and same_surface(
                body.faces[adj[0]].surface, body.faces[adj[1]].surface, 1e-9 * max(extent, 1.0)
            ):
                continue
            on_cut = any(fi in caps for fi in adj)
            pts = t.positions[t.edge_samples[ei]]
            lines.append((pts, on_cut, s.role))
        for pts in _silhouettes(body, t, view.direction, 1e-9 * max(extent, 1.0)):
            lines.append((pts, False, s.role))
    rays = 0
    for pts, on_cut, role in lines:
        layer = role
        if on_cut:
            drawing.line(view.project(pts), "section", layer)
            continue
        dense, orig = _densify(pts, step)
        mids = 0.5 * (dense[1:] + dense[:-1])
        rays += len(mids)
        if rays > MAX_RAYS:
            raise CadError("LIMIT_EXCEEDED", "the drawing needs too many visibility rays")
        vis = _visible(accel, mids - center, view.direction, extent)
        start = 0
        for k in range(1, len(vis) + 1):
            if k == len(vis) or vis[k] != vis[start]:
                # The edge's own samples (on the exact curve) and the split points.
                keep = orig[start : k + 1].copy()
                keep[0] = keep[-1] = True
                piece = dense[start : k + 1][keep]
                if vis[start]:
                    drawing.line(view.project(piece), "visible", layer)
                elif hidden:
                    drawing.line(view.project(piece), "hidden", layer)
                start = k
    drawing.info["solids"] = len(kept)
    drawing.info["visibility_rays"] = rays
    drawing.info["chord_m"] = chord
    drawing.info["visibility_step_m"] = step
    return drawing


def _loop_points(body: Body, t, loop) -> np.ndarray:
    out = []
    for ei, fwd in loop:
        ids = t.edge_samples[ei]
        ids = ids if fwd else ids[::-1]
        out.append(t.positions[ids[:-1]])
    return np.concatenate(out)


def _densify(pts: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points at most ``step`` apart along the polyline, and which are its own."""
    out = [pts[:1]]
    own = [np.array([True])]
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        ts = np.linspace(0.0, 1.0, n + 1)[1:]
        out.append(a[None] + ts[:, None] * (b - a)[None])
        mask = np.zeros(n, dtype=bool)
        mask[-1] = True
        own.append(mask)
    return np.concatenate(out), np.concatenate(own)


def _silhouettes(body: Body, t, direction: np.ndarray, tol: float) -> list[np.ndarray]:
    """Polylines where curved surfaces turn from facing the viewer to facing away (across
    the seams between faces of one surface, too: a cylinder's halves)."""
    p = t.positions
    tri = t.triangles
    normal = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    facing = normal @ (-direction) > 0
    curved = np.array([f.surface.kind != "plane" for f in body.faces])
    rows = np.flatnonzero(curved[t.face_of])
    edges: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        a, b, c = tri[r]
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(int(r))
    same: dict[tuple[int, int], bool] = {}
    segs = []
    for k, rs in edges.items():
        if len(rs) != 2 or facing[rs[0]] == facing[rs[1]]:
            continue
        f1, f2 = int(t.face_of[rs[0]]), int(t.face_of[rs[1]])
        if f1 != f2:
            key = (min(f1, f2), max(f1, f2))
            if key not in same:
                same[key] = same_surface(body.faces[f1].surface, body.faces[f2].surface, tol)
            if not same[key]:
                continue  # a real edge between surfaces: drawn as an edge
        segs.append(k)
    return _chain(segs, p)


def _chain(segs, p) -> list[np.ndarray]:
    nb: dict[int, list[int]] = {}
    for a, b in segs:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    used: set[tuple[int, int]] = set()
    out = []
    ends = [v for v, n in nb.items() if len(n) != 2] + list(nb)
    for start in ends:
        for nxt in nb[start]:
            if (min(start, nxt), max(start, nxt)) in used:
                continue
            path = [start]
            prev, cur = start, nxt
            used.add((min(prev, cur), max(prev, cur)))
            path.append(cur)
            while len(nb[cur]) == 2:
                a, b = nb[cur]
                n2 = b if a == prev else a
                key = (min(cur, n2), max(cur, n2))
                if key in used:
                    break
                used.add(key)
                prev, cur = cur, n2
                path.append(cur)
            out.append(p[np.array(path)])
    return out


def _accel(tess, center):
    from ...rendering.raytrace.accel import BLASCache, SceneAccel
    from ...rendering.scene_data import RenderMesh

    meshes = []
    insts = []
    for k, t in enumerate(tess):
        pos = np.ascontiguousarray((t.positions - center).astype(np.float32))
        tris = np.ascontiguousarray(t.triangles.astype(np.uint32))
        pos.setflags(write=False)
        tris.setflags(write=False)
        meshes.append(RenderMesh(pos, tris, None))
        insts.append(types.SimpleNamespace(mesh=k, world=np.eye(4)))
    rs = types.SimpleNamespace(meshes=tuple(meshes), instances=tuple(insts))
    return SceneAccel.build(rs, cache=BLASCache(budget=512 * 2**20))


def _visible(accel, pts: np.ndarray, direction: np.ndarray, extent: float) -> np.ndarray:
    """Whether each point sees the viewer (orthographic: along -direction)."""
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    eps = 2e-5 * max(extent, 1e-6)
    d = np.broadcast_to(-direction, (n, 3)).astype(np.float64).copy()
    o = pts.astype(np.float64)
    occluded = accel.occluded(o, d, np.full(n, eps), np.full(n, np.inf))
    return ~occluded
