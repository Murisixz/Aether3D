"""Spatial queries on a scene: bounds, discovery, rays, distances, containment, overlaps,
regions, cameras, lines of sight and relations. ``docs/SPATIAL.md`` describes the
conventions; :class:`~aether3d.Scene` exposes every query as a method.

All queries read the scene's :class:`~aether3d.spatial.index.SpatialIndex` (cached
per scene revision) and change nothing: no history step, no revision change.

**Which objects take part.** Objects with geometry -- meshes and procedural
objects, evaluated -- in world space. Hidden objects (``obj.visible == False``)
are left out unless ``include_hidden=True``; an object named as the subject of
a query (``bounds``, ``contains``, ``distance``, ``relation``, the target of
``visible_from``) always takes part. ``objects`` restricts a query to some
objects and ``exclude`` leaves some out (objects, ids or names, or lists of
them).

**Exactness.** Every answer comes from the triangles themselves, in float64
world coordinates computed from the meshes' float32 positions. Bounding boxes
only select candidates; results that rest on boxes alone say so (``exact``
false, relation ``"bounds"``).
"""

from __future__ import annotations

import fnmatch
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError, QueryError
from . import geometry as geo
from .index import Entry, SpatialIndex, _as_list, get_index
from .rays import MERGE_TOLERANCE, RayHits, all_hits, merge_coincident, rendered_filter
from .rays import rendered_scene as _rendered_scene
from .types import (
    AABB,
    CameraRay,
    Containment,
    LineOfSight,
    ObjectBounds,
    ObjectDistance,
    ObjectRef,
    OrientedBox,
    Overlap,
    RegionMatch,
    Relation,
    ScreenPoint,
    Sphere,
    SurfaceHit,
    ViewMatch,
    Visibility,
    vec3,
)

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

SURFACES = ("geometry", "rendered")
"""``"geometry"``: every triangle stops rays from both sides. ``"rendered"``: rays pass
where the renderers draw nothing (alpha-mask cut-outs, backs of single-sided materials)."""
KINDS = ("mesh", "procedural", "camera", "light", "empty")
REGION_MODES = ("intersects", "inside", "bounds")
CULL_MODES = ("none", "back", "front")
DEFAULT_TOLERANCE = 1e-6
"""World units: how close counts as touching for containment and relations."""
DEFAULT_MARGIN = 1e-6
"""Lines of sight ignore surfaces this close to their ends (relative to their length)."""
WINDING_BUDGET = 20_000_000
"""Point-triangle pairs above which point-in-solid tests count ray crossings instead of
summing solid angles (both exact away from the surface)."""
PAIR_CHUNK = 65_536
"""Triangle pairs tested at once by distance and intersection queries."""
MAX_SAMPLES = 4096
CROSSING_DIRECTIONS = np.array(
    [
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
        [0.2672612419124244, -0.5345224838248488, 0.8017837257372732],
        [-0.8164965809277261, 0.4082482904638631, 0.4082482904638631],
        [0.1301889109, 0.9861607208, -0.1028838017],
        [-0.3713906764, -0.5570860145, -0.7427813527],
        [0.9128709292, -0.1825741858, 0.3651483717],
    ]
)


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #


def _number(value: Any, what: str, *, minimum: float | None = None, inf: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise QueryError("INVALID_ARGUMENT", f"must be a number, got {value!r}", what)
    v = float(value)
    if math.isnan(v) or (math.isinf(v) and not (inf and v > 0)):
        raise QueryError("INVALID_ARGUMENT", f"must be a finite number, got {value!r}", what)
    if minimum is not None and v < minimum:
        raise QueryError("INVALID_ARGUMENT", f"must be at least {minimum:g}, got {value!r}", what)
    return v


def _integer(value: Any, what: str, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise QueryError("INVALID_ARGUMENT", f"must be an integer, got {value!r}", what)
    v = int(value)
    if v < minimum or (maximum is not None and v > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise QueryError("INVALID_ARGUMENT", f"must be {bound}, got {value!r}", what)
    return v


def _flag(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise QueryError("INVALID_ARGUMENT", f"must be true or false, got {value!r}", what)
    return value


def _choice(value: Any, what: str, options: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in options:
        raise QueryError(
            "INVALID_ARGUMENT", f"must be one of {', '.join(options)}, got {value!r}", what
        )
    return value


def _unit(direction: Any, what: str = "direction") -> np.ndarray:
    d = np.asarray(vec3(direction, what))
    length = float(np.linalg.norm(d))
    if not length > 0.0 or not math.isfinite(length):
        raise QueryError("INVALID_ARGUMENT", "must not be the zero vector", what)
    return d / length


def _ray_range(min_distance: Any, max_distance: Any) -> tuple[float, float]:
    lo = _number(min_distance, "min_distance", minimum=0.0)
    hi = _number(max_distance, "max_distance", minimum=0.0, inf=True)
    if not hi > lo:
        raise QueryError(
            "INVALID_ARGUMENT", f"must exceed min_distance ({lo:g}), got {hi:g}", "max_distance"
        )
    return lo, hi


def _region(region: Any) -> AABB | Sphere:
    if isinstance(region, (AABB, Sphere)):
        return region
    raise QueryError(
        "INVALID_ARGUMENT",
        f"must be a Box (AABB) or a Sphere, got {type(region).__name__}",
        "region",
        hint="aether3d.spatial.Box((x0, y0, z0), (x1, y1, z1)) or Sphere((x, y, z), r)",
    )


def _t3(v: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in v)
    return (x, y, z)


def _ref(obj: Any) -> ObjectRef:
    return ObjectRef(obj.id, obj.name)


def _entries(
    index: SpatialIndex, objects: Any, exclude: Any, include_hidden: Any, surfaces: str
) -> list[Entry]:
    include_hidden = _flag(include_hidden, "include_hidden")
    if surfaces == "rendered" and include_hidden:
        raise QueryError(
            "INVALID_ARGUMENT",
            "hidden objects do not render, so rendered surfaces cannot include them",
            "include_hidden",
            hint='use surfaces="geometry" to query hidden objects',
        )
    return index.select(objects, exclude, include_hidden)


def _point_objects(index: SpatialIndex, objects: Any, exclude: Any, include_hidden: bool):
    """Objects without geometry (cameras, lights, empties): points at their world origin."""
    scene = index.scene
    chosen = [o for o in scene if o.id not in index.by_id and o.id not in index.failed]
    if objects is not None:
        ids = {index.resolve(o, "objects").id for o in _as_list(objects)}
        chosen = [o for o in chosen if o.id in ids]
    if exclude is not None:
        drop = {index.resolve(o, "exclude").id for o in _as_list(exclude)}
        chosen = [o for o in chosen if o.id not in drop]
    if not include_hidden:
        chosen = [o for o in chosen if o.visible]
    worlds = index.worlds
    return [(o, worlds[o.id][:3, 3].copy()) for o in chosen if np.all(np.isfinite(worlds[o.id]))]


# --------------------------------------------------------------------------- #
# Surface points
# --------------------------------------------------------------------------- #


def _hit(
    index: SpatialIndex,
    entry: Entry,
    tri: int,
    bary: np.ndarray,
    distance: float,
    direction: np.ndarray | None = None,
) -> SurfaceHit:
    mesh = entry.mesh
    tri = int(tri)
    tv = mesh.triangles[tri].astype(np.int64)
    corners = entry.vertices[tv]
    b = np.asarray(bary, dtype=np.float64)
    position = b[0] * corners[0] + b[1] * corners[1] + b[2] * corners[2]
    g = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    length = float(np.linalg.norm(g))
    normal = g / length * entry.orientation if length > 0.0 else np.zeros(3)
    shading = normal
    if mesh.normals is not None and not entry.singular:
        nv = mesh.normals[tv].astype(np.float64) @ entry.normal_matrix.T
        lengths = np.linalg.norm(nv, axis=1, keepdims=True)
        nv = np.divide(nv, lengths, out=np.zeros_like(nv), where=lengths > 1e-300)
        ns = b[0] * nv[0] + b[1] * nv[1] + b[2] * nv[2]
        ns_length = float(np.linalg.norm(ns))
        if ns_length > 1e-6:
            shading = ns / ns_length
    front = None if direction is None else bool(float(direction @ normal) < 0.0)
    group = int(mesh.groups[tri]) if mesh.groups is not None else 0
    uv = None
    if mesh.uvs is not None:
        u = b @ mesh.uvs[tv].astype(np.float64)
        uv = (float(u[0]), float(u[1]))
    material = entry.obj.material_for_group(group)
    users = index.mesh_users(entry)
    return SurfaceHit(
        _ref(entry.obj),
        _t3(position),
        float(distance),
        _t3(normal),
        _t3(shading),
        front,
        tri,
        group,
        _t3(b),
        uv,
        None if material is None else ObjectRef(material.id, material.name),
        _ref(users[0].obj),
        len(users),
    )


# --------------------------------------------------------------------------- #
# Rays
# --------------------------------------------------------------------------- #


def _trace(
    index: SpatialIndex,
    entries: list[Entry],
    o: np.ndarray,
    d: np.ndarray,
    tmin: np.ndarray,
    tmax: np.ndarray,
    surfaces: str,
    cull: str = "none",
    first: bool = False,
) -> tuple[RayHits, list[Entry]]:
    """Hits of rays ``o + t d`` (``d`` unit) with ``tmin < t < tmax``, sorted by ray, distance,
    scene order and triangle; with ``first`` only the first hit of every ray.

    The closest hit comes from the closest-hit traversal; all hits up to just past
    it are then collected and ordered exactly like :func:`raycast_all`, so the
    first hit is the same either way.
    """
    n = o.shape[0]
    tmin = np.broadcast_to(np.asarray(tmin, dtype=np.float64), (n,)).copy()
    tmax = np.broadcast_to(np.asarray(tmax, dtype=np.float64), (n,)).copy()
    if surfaces == "rendered":
        ps, ents = _rendered_scene(index, entries)
        accel = ps.accel
    else:
        ps = None
        accel, ents = index.accel(entries)
    if not ents:
        return RayHits.empty(), ents
    scale = np.abs(o).max(axis=1)
    rays = np.arange(n)
    if first:
        if ps is None:
            closest = accel.closest_hit(o, d, tmin, tmax)
        else:
            closest, _ = ps.trace(o, d, tmin, tmax, cull=cull)
        rays = np.flatnonzero(closest.inst >= 0)
        if not rays.size:
            return RayHits.empty(), ents
        t = closest.t[rays]
        window = t + 4.0 * MERGE_TOLERANCE * (1.0 + np.abs(t) + scale[rays])
        tmax = tmax.copy()
        tmax[rays] = np.minimum(tmax[rays], window)
    hits = all_hits(accel, o[rays], d[rays], tmin[rays], tmax[rays])
    hits.ray = rays[hits.ray]
    if ps is not None:
        hits = rendered_filter(ps, hits, o, d, cull)
    hits = merge_coincident(hits, scale)
    order = np.array([e.order for e in ents], dtype=np.int64)
    hits = hits.take(np.lexsort((hits.prim, order[hits.inst], hits.t, hits.ray)))
    if first and len(hits):
        lead = np.ones(len(hits), dtype=bool)
        lead[1:] = hits.ray[1:] != hits.ray[:-1]
        hits = hits.take(np.flatnonzero(lead))
    return hits, ents


def _raycast(
    scene: Scene,
    origin: Any,
    direction: Any,
    max_distance: Any,
    min_distance: Any,
    objects: Any,
    exclude: Any,
    include_hidden: Any,
    surfaces: Any,
    limit: int | None,
    first: bool,
) -> list[SurfaceHit]:
    o = np.asarray(vec3(origin, "origin"))
    d = _unit(direction)
    lo, hi = _ray_range(min_distance, max_distance)
    surfaces = _choice(surfaces, "surfaces", SURFACES)
    if limit is not None:
        limit = _integer(limit, "limit")
    index = get_index(scene)
    entries = _entries(index, objects, exclude, include_hidden, surfaces)
    hits, ents = _trace(index, entries, o[None], d[None], lo, hi, surfaces, first=first)
    rows = range(len(hits)) if limit is None else range(min(limit, len(hits)))
    return [_hit(index, ents[hits.inst[k]], hits.prim[k], hits.b[k], hits.t[k], d) for k in rows]


def raycast(
    scene: Scene,
    origin: Any,
    direction: Any,
    *,
    max_distance: float = math.inf,
    min_distance: float = 0.0,
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
    surfaces: str = "geometry",
) -> SurfaceHit | None:
    """The first surface along a ray (``None``: nothing in range).

    ``direction`` need not be unit length; distances are measured along the
    normalized ray, and only hits with ``min_distance < distance < max_distance``
    count (a surface through the origin itself is not hit). Triangles stop rays
    from both sides; ``surfaces="rendered"`` lets rays through what the
    renderers do not draw (alpha-mask cut-outs, backs of single-sided
    materials; visible objects only). The result is the first hit of
    :func:`raycast_all`.
    """
    found = _raycast(
        scene,
        origin,
        direction,
        max_distance,
        min_distance,
        objects,
        exclude,
        include_hidden,
        surfaces,
        None,
        True,
    )
    return found[0] if found else None


def raycast_all(
    scene: Scene,
    origin: Any,
    direction: Any,
    *,
    max_distance: float = math.inf,
    min_distance: float = 0.0,
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
    surfaces: str = "geometry",
    limit: int | None = None,
) -> list[SurfaceHit]:
    """Every surface along a ray, nearest first (ties: scene order, then triangle index).

    A ray through an edge or vertex shared by several triangles of one object
    reports one hit there (the lowest triangle index). ``limit`` keeps the first
    hits only. Other arguments as for :func:`raycast`.
    """
    return _raycast(
        scene,
        origin,
        direction,
        max_distance,
        min_distance,
        objects,
        exclude,
        include_hidden,
        surfaces,
        limit,
        False,
    )


# --------------------------------------------------------------------------- #
# Nearest points
# --------------------------------------------------------------------------- #


def _leaf_items(bvh, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Primitives of leaf ``nodes`` and, for each, the position of its node in ``nodes``."""
    first, count = bvh.first[nodes], bvh.count[nodes]
    total = int(count.sum())
    seg = np.cumsum(count) - count
    owner = np.repeat(np.arange(nodes.size), count)
    return bvh.order[first[owner] + (np.arange(total) - seg[owner])], owner


def _slack(value: float) -> float:
    return value * (1.0 + 1e-12) + 1e-300


def _nearest(entry: Entry, p: np.ndarray, bound: float = math.inf):
    """``(d2, triangle, point, barycentrics)`` of the entry's surface nearest to ``p`` if within
    distance ``bound``, else ``None``. Ties: the lowest triangle index.

    A best-first descent of the entry's BVH: nodes whose box is farther than the
    current bound are skipped, and every box's farthest point bounds the distance
    to the triangles inside it.
    """
    bvh = entry.bvh
    if bvh.order.size == 0:
        return None
    lo, hi = entry.node_boxes
    best: tuple | None = None
    nodes = np.zeros(1, dtype=np.int64)
    near2, far2 = geo.box_point_distances(p, lo[nodes], hi[nodes])
    bound2 = _slack(bound * bound)
    limit = min(bound2, float(far2.min()))
    while nodes.size:
        keep = near2 <= _slack(limit)
        nodes = nodes[keep]
        if not nodes.size:
            break
        leaf = bvh.child[nodes, 0] < 0
        if leaf.any():
            tris, _ = _leaf_items(bvh, nodes[leaf])
            a, b, c = entry.corners(tris)
            pts, bary = geo.closest_points_on_triangles(p, a, b, c)
            diff = pts - p
            d2 = np.einsum("ij,ij->i", diff, diff)
            k = np.lexsort((tris, d2))[0]
            if best is None or (d2[k], tris[k]) < (best[0], best[1]):
                best = (float(d2[k]), int(tris[k]), pts[k], bary[k])
                limit = min(limit, best[0])
        inner = nodes[~leaf]
        if not inner.size:
            break
        nodes = bvh.child[inner].reshape(-1)
        near2, far2 = geo.box_point_distances(p, lo[nodes], hi[nodes])
        limit = min(limit, float(far2.min()))
    if best is None or math.sqrt(best[0]) > bound:
        return None
    return best


def closest_point(
    scene: Scene,
    point: Any,
    *,
    max_distance: float = math.inf,
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
) -> SurfaceHit | None:
    """The surface point nearest to ``point`` (``None``: no surface within ``max_distance``).

    ``distance`` is the point-to-surface distance. Ties (equal distances): the
    first object in scene order, then the lowest triangle index.
    """
    found = nearest_objects(
        scene,
        point,
        count=1,
        max_distance=max_distance,
        objects=objects,
        exclude=exclude,
        include_hidden=include_hidden,
    )
    return found[0] if found else None


def nearest_objects(
    scene: Scene,
    point: Any,
    *,
    count: int | None = None,
    max_distance: float = math.inf,
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
) -> list[SurfaceHit]:
    """Objects by the distance of their surface from ``point``, nearest first (each with its
    nearest surface point); at most ``count`` of them, within ``max_distance``."""
    p = np.asarray(vec3(point, "point"))
    limit = _number(max_distance, "max_distance", minimum=0.0, inf=True)
    if count is not None:
        count = _integer(count, "count")
    index = get_index(scene)
    entries = index.select(objects, exclude, _flag(include_hidden, "include_hidden"))
    if not entries:
        return []
    lo = np.array([e.box[0] for e in entries])
    hi = np.array([e.box[1] for e in entries])
    near2, _ = geo.box_point_distances(p, lo, hi)
    bound = limit
    found: list[tuple[float, int, Entry, tuple]] = []
    for k in np.lexsort((np.arange(len(entries)), near2)):
        if near2[k] > _slack(bound * bound):
            break
        entry = entries[k]
        best = _nearest(entry, p, bound)
        if best is None:
            continue
        found.append((best[0], entry.order, entry, best))
        if count is not None:
            found.sort(key=lambda f: (f[0], f[1]))
            del found[count:]
            if len(found) == count:
                bound = min(bound, math.sqrt(found[-1][0]))
    found.sort(key=lambda f: (f[0], f[1]))
    return [_hit(index, entry, best[1], best[3], math.sqrt(best[0])) for _, _, entry, best in found]


# --------------------------------------------------------------------------- #
# Containment
# --------------------------------------------------------------------------- #


def _winding(entry: Entry, points: np.ndarray) -> np.ndarray:
    """Generalized winding numbers of the entry's world surface at ``points`` (sign-corrected
    for mirroring transforms: positive inside an outward-facing mesh)."""
    a, b, c = entry.corners()
    return geo.winding_numbers(points, a, b, c) * entry.orientation


def _inside_solid(index: SpatialIndex, entry: Entry, points: np.ndarray) -> np.ndarray:
    """Whether points off the surface are inside a closed entry's solid.

    Few points: the winding number. Many: the parity of ray crossings (Stage 8
    traversal), trying further fixed directions when a ray meets an edge, a vertex
    or a surface edge-on; rays that stay ambiguous use the winding number.
    """
    m = points.shape[0]
    tris = entry.mesh.triangle_count
    if m * tris <= WINDING_BUDGET or entry.singular:
        return np.abs(_winding(entry, points)) > 0.5
    accel, _ = index.accel([entry])
    result = np.zeros(m, dtype=bool)
    pending = np.arange(m)
    for direction in CROSSING_DIRECTIONS:
        if not pending.size:
            break
        o = points[pending]
        d = np.broadcast_to(direction, o.shape).copy()
        hits = all_hits(accel, o, d, np.zeros(o.shape[0]), np.full(o.shape[0], np.inf))
        a, b, c = entry.corners(hits.prim)
        n = np.cross(b - a, c - a)
        cos = np.abs(np.einsum("ij,ij->i", n, d[hits.ray])) / np.maximum(
            np.linalg.norm(n, axis=1), 1e-300
        )
        bad = (hits.b.min(axis=1) < 1e-9) | (cos < 1e-9)
        ambiguous = np.zeros(o.shape[0], dtype=bool)
        ambiguous[hits.ray[bad]] = True
        crossings = np.bincount(hits.ray, minlength=o.shape[0])
        done = ~ambiguous
        result[pending[done]] = crossings[done] % 2 == 1
        pending = pending[ambiguous]
    if pending.size:
        result[pending] = np.abs(_winding(entry, points[pending])) > 0.5
    return result


def _undefined_reason(topology: dict[str, Any]) -> str | None:
    if topology["solid"]:
        return None
    reasons = []
    if not topology["closed"]:
        reasons.append("open: it has boundary edges")
    if not topology["edge_manifold"]:
        reasons.append("non-manifold: edges shared by more than two triangles")
    if not topology["consistently_oriented"]:
        reasons.append("inconsistently oriented triangles")
    return "; ".join(reasons) + " -- it bounds no well-defined solid"


def contains_point(
    scene: Scene, obj: Any, point: Any, *, tolerance: float = DEFAULT_TOLERANCE
) -> Containment:
    """Whether ``point`` is inside ``obj``'s solid (see :class:`~aether3d.spatial.Containment`).

    Points within ``tolerance`` of the surface are ``"surface"``. Inside/outside
    is decided by the winding number, and only for closed meshes (watertight,
    edge-manifold, consistently oriented; vertices at equal positions welded);
    for other meshes the state is ``"undefined"`` and ``winding`` (fractional)
    measures how much of the surface surrounds the point.
    """
    p = np.asarray(vec3(point, "point"))
    tol = _number(tolerance, "tolerance", minimum=0.0)
    index = get_index(scene)
    entry = index.entry(obj, "object")
    topology = index.topology(entry)
    nearest = _nearest(entry, p)
    distance = math.sqrt(nearest[0]) if nearest is not None else math.inf
    winding = float(_winding(entry, p[None])[0])
    if distance <= tol:
        state = "surface"
    elif not topology["solid"]:
        state = "undefined"
    else:
        state = "inside" if abs(winding) > 0.5 else "outside"
    return Containment(
        _ref(entry.obj),
        _t3(p),
        state,
        winding,
        distance,
        bool(topology["solid"]),
        _undefined_reason(topology) if state == "undefined" else None,
    )


# --------------------------------------------------------------------------- #
# Pairs of objects: distance and intersection
# --------------------------------------------------------------------------- #


def _pair_triangles(ea: Entry, eb: Entry, na: np.ndarray, nb: np.ndarray):
    """All triangle pairs of the leaf pairs ``(na[k], nb[k])``."""
    A, B = ea.bvh, eb.bvh
    ca, cb = A.count[na], B.count[nb]
    per = ca * cb
    total = int(per.sum())
    owner = np.repeat(np.arange(na.size), per)
    within = np.arange(total) - np.repeat(np.cumsum(per) - per, per)
    ia = within // cb[owner]
    ib = within % cb[owner]
    return A.order[A.first[na][owner] + ia], B.order[B.first[nb][owner] + ib]


def _pair_walk(ea: Entry, eb: Entry, visit) -> None:
    """Walk pairs of BVH nodes of two entries, nearest boxes first.

    ``visit(near, far, None, na, nb)`` sees every batch of node pairs (``near``
    and ``far``: the least and greatest distances between their boxes) and
    returns ``(limit, stop)``: pairs farther apart than ``limit`` are dropped,
    ``stop`` ends the walk. ``visit(None, None, (ta, tb), na, nb)`` receives the
    triangle pairs of leaf/leaf node pairs.
    """
    A, B = ea.bvh, eb.bvh
    if A.order.size == 0 or B.order.size == 0:
        return
    alo, ahi = ea.node_boxes
    blo, bhi = eb.node_boxes
    stack = [(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))]
    while stack:
        na, nb = stack.pop()
        near, far = geo.box_pair_distances(alo[na], ahi[na], blo[nb], bhi[nb])
        limit, stop = visit(near, far, None, na, nb)
        if stop:
            return
        keep = near <= _slack(limit)
        na, nb, near = na[keep], nb[keep], near[keep]
        if not na.size:
            continue
        la = A.child[na, 0] < 0
        lb = B.child[nb, 0] < 0
        both = la & lb
        if both.any():
            sel = np.flatnonzero(both)
            for s in range(0, sel.size, max(1, PAIR_CHUNK // 16)):
                part = sel[s : s + max(1, PAIR_CHUNK // 16)]
                ta, tb = _pair_triangles(ea, eb, na[part], nb[part])
                _, stop = visit(None, None, (ta, tb), na[part], nb[part])
                if stop:
                    return
        rest = ~both
        if not rest.any():
            continue
        size_a = np.einsum("ij,ij->i", ahi[na] - alo[na], ahi[na] - alo[na])
        size_b = np.einsum("ij,ij->i", bhi[nb] - blo[nb], bhi[nb] - blo[nb])
        split_a = rest & ~la & (lb | (size_a >= size_b))
        split_b = rest & ~split_a
        new_a = np.concatenate([A.child[na[split_a], 0], A.child[na[split_a], 1], na[split_b]])
        new_b = np.concatenate([nb[split_a], nb[split_a], B.child[nb[split_b], 0]])
        new_a = np.concatenate([new_a, na[split_b]])
        new_b = np.concatenate([new_b, B.child[nb[split_b], 1]])
        # Farther pairs first onto the stack, so the nearest are taken next.
        near_new, _ = geo.box_pair_distances(alo[new_a], ahi[new_a], blo[new_b], bhi[new_b])
        order = np.argsort(-near_new, kind="stable")
        new_a, new_b = new_a[order], new_b[order]
        step = PAIR_CHUNK
        for s in range(0, new_a.size, step):
            stack.append((new_a[s : s + step], new_b[s : s + step]))


def _object_distance(ea: Entry, eb: Entry):
    """``(distance, point_a, point_b, intersecting)`` between two entries' surfaces."""
    state: dict[str, Any] = {"best": math.inf, "ub": math.inf, "pair": None}

    def visit(near, far, pairs, na, nb):
        if pairs is None:
            if far is not None and far.size:
                state["ub"] = min(state["ub"], float(far.min()))
            return min(state["ub"], state["best"]), False
        ta, tb = pairs
        a1, b1, c1 = ea.corners(ta)
        a2, b2, c2 = eb.corners(tb)
        dist, p1, p2 = geo.triangle_distances(a1, b1, c1, a2, b2, c2)
        k = np.lexsort((tb, ta, dist))[0]
        if state["pair"] is None or (dist[k], ta[k], tb[k]) < state["pair"][0]:
            state["pair"] = ((float(dist[k]), int(ta[k]), int(tb[k])), p1[k], p2[k])
            state["best"] = float(dist[k])
        return state["best"], state["best"] == 0.0

    _pair_walk(ea, eb, visit)
    if state["pair"] is None:
        return math.inf, np.zeros(3), np.zeros(3), False
    (dist, _, _), p1, p2 = state["pair"]
    return dist, p1, p2, dist == 0.0


def _surfaces_meet(ea: Entry, eb: Entry, tolerance: float) -> bool:
    """Whether the surfaces share a point (or come within ``tolerance``)."""
    state = {"hit": False}

    def visit(near, far, pairs, na, nb):
        if pairs is None:
            return tolerance, False
        ta, tb = pairs
        a1, b1, c1 = ea.corners(ta)
        a2, b2, c2 = eb.corners(tb)
        if tolerance > 0.0:
            dist, _, _ = geo.triangle_distances(a1, b1, c1, a2, b2, c2)
            hit = bool((dist <= tolerance).any())
        else:
            hit = bool(geo.triangles_intersect(a1, b1, c1, a2, b2, c2).any())
        state["hit"] = hit
        return tolerance, hit

    lo1, hi1 = ea.box
    lo2, hi2 = eb.box
    if np.any(lo1 > hi2 + tolerance) or np.any(lo2 > hi1 + tolerance):
        return False
    _pair_walk(ea, eb, visit)
    return state["hit"]


def _all_inside(index: SpatialIndex, inner: Entry, outer: Entry) -> bool:
    """With surfaces that do not meet: whether ``inner`` lies inside ``outer``'s closed solid
    (one vertex of every connected component of ``inner`` decides for it)."""
    if not index.topology(outer)["solid"]:
        return False
    lo1, hi1 = inner.box
    lo2, hi2 = outer.box
    if np.any(lo1 < lo2) or np.any(hi1 > hi2):
        return False
    reps = inner.vertices[index.representatives(inner)]
    return bool(_inside_solid(index, outer, reps).all())


def distance(scene: Scene, a: Any, b: Any) -> ObjectDistance:
    """The shortest distance between two objects' surfaces and where it is reached.

    Exact (triangle pairs, pruned with both objects' BVHs); zero when the
    surfaces touch or cross (``intersecting``; both points are then one common
    point). An object inside another's solid without touching it has a positive
    distance: see :func:`relation` for containment.
    """
    index = get_index(scene)
    ea = index.entry(a, "a")
    eb = index.entry(b, "b")
    if ea is eb:
        raise QueryError("INVALID_ARGUMENT", "a and b are the same object", "b")
    dist, p1, p2, hit = _object_distance(ea, eb)
    return ObjectDistance(_ref(ea.obj), _ref(eb.obj), dist, _t3(p1), _t3(p2), hit)


def overlaps(
    scene: Scene,
    objects: Any = None,
    *,
    against: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
    exact: bool = True,
    tolerance: float = 0.0,
) -> list[Overlap]:
    """Pairs of objects that overlap: among ``objects`` (default: all), or between
    ``objects`` and ``against``.

    Broad phase: world boxes (grown by ``tolerance``) that overlap. With
    ``exact`` (default) every candidate is decided on its triangles: the pair is
    reported when the surfaces intersect or come within ``tolerance``
    (``intersecting``), or when one lies entirely inside the other's closed solid
    (``contained``). With ``exact=False`` the broad-phase candidates are returned
    as they are (``exact`` false, ``intersecting`` ``None``): overlapping bounds,
    not collisions. Pairs are ordered by scene order.
    """
    tol = _number(tolerance, "tolerance", minimum=0.0)
    exact = _flag(exact, "exact")
    include_hidden = _flag(include_hidden, "include_hidden")
    index = get_index(scene)
    first = index.select(objects, exclude, include_hidden)
    second = first if against is None else index.select(against, exclude, include_hidden)
    if not first or not second:
        return []
    lo_b = np.array([e.box[0] for e in second]) - tol
    hi_b = np.array([e.box[1] for e in second]) + tol
    from ..rendering.raytrace.bvh import build_bvh

    tree = build_bvh(lo_b, hi_b)
    lo_a = np.array([e.box[0] for e in first])
    hi_a = np.array([e.box[1] for e in first])
    qa, qb = geo.bvh_box_pairs(tree, lo_b, hi_b, lo_a, hi_a)
    pairs: set[tuple[int, int]] = set()
    by_order: dict[int, Entry] = {}
    for i, j in zip(qa.tolist(), qb.tolist(), strict=True):
        ea, eb = first[i], second[j]
        if ea is eb:
            continue
        by_order[ea.order], by_order[eb.order] = ea, eb
        pairs.add((min(ea.order, eb.order), max(ea.order, eb.order)))
    out: list[Overlap] = []
    for oa, ob in sorted(pairs):
        ea, eb = by_order[oa], by_order[ob]
        if not exact:
            out.append(Overlap(_ref(ea.obj), _ref(eb.obj), True, None, None, False))
            continue
        if _surfaces_meet(ea, eb, tol):
            out.append(Overlap(_ref(ea.obj), _ref(eb.obj), True, True, None, True))
        elif _all_inside(index, ea, eb):
            out.append(Overlap(_ref(ea.obj), _ref(eb.obj), True, False, "a_in_b", True))
        elif _all_inside(index, eb, ea):
            out.append(Overlap(_ref(ea.obj), _ref(eb.obj), True, False, "b_in_a", True))
    return out


# --------------------------------------------------------------------------- #
# Regions
# --------------------------------------------------------------------------- #


def _region_box(region: AABB | Sphere) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(region, AABB):
        return np.asarray(region.min), np.asarray(region.max)
    c = np.asarray(region.center)
    return c - region.radius, c + region.radius


def _points_in_region(region: AABB | Sphere, pts: np.ndarray) -> np.ndarray:
    if isinstance(region, AABB):
        return np.all((pts >= region.min) & (pts <= region.max), axis=1)
    diff = pts - np.asarray(region.center)
    return np.einsum("ij,ij->i", diff, diff) <= region.radius * region.radius


def _surface_meets_region(entry: Entry, region: AABB | Sphere) -> bool:
    rlo, rhi = _region_box(region)
    _, tris = geo.bvh_box_pairs(entry.bvh, *entry.triangle_boxes, rlo[None], rhi[None])
    if not tris.size:
        return False
    a, b, c = entry.corners(tris)
    if isinstance(region, AABB):
        verts = geo.box_vertices(rlo, rhi)
        return bool(
            geo.triangles_intersect_convex(a, b, c, verts, geo.BOX_NORMALS, geo.BOX_NORMALS).any()
        )
    d2, _ = geo.point_triangle_distances(np.asarray(region.center), a, b, c)
    return bool((d2 <= region.radius * region.radius).any())


def _region_relation(index: SpatialIndex, entry: Entry, region: AABB | Sphere, mode: str):
    rlo, rhi = _region_box(region)
    lo, hi = entry.box
    if np.any(lo > rhi) or np.any(hi < rlo):
        return None
    if isinstance(region, Sphere):  # the sphere itself must reach the box
        near2, _ = geo.box_point_distances(np.asarray(region.center), lo[None], hi[None])
        if near2[0] > region.radius * region.radius:
            return None
    if mode == "bounds":
        return "bounds"
    if _points_in_region(region, entry.points).all():
        return "inside"
    if mode == "inside":
        return None
    if _surface_meets_region(entry, region):
        return "intersects"
    center = np.asarray(region.center)[None]
    if index.topology(entry)["solid"] and _inside_solid(index, entry, center)[0]:
        return "contains"
    return None


def query_region(
    scene: Scene,
    region: Any,
    *,
    mode: str = "intersects",
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
) -> list[RegionMatch]:
    """Objects in a region (a :class:`Box` or :class:`Sphere`, closed), in scene order.

    ``mode="intersects"`` (default): objects whose geometry is inside the region
    (relation ``"inside"``), meets it (``"intersects"``), or encloses it within a
    closed solid (``"contains"``). ``mode="inside"``: only those entirely inside.
    ``mode="bounds"``: objects whose world box meets the region (``"bounds"``,
    candidates only). Objects without geometry (cameras, lights, empties) count as
    their world position.
    """
    region = _region(region)
    mode = _choice(mode, "mode", REGION_MODES)
    include_hidden = _flag(include_hidden, "include_hidden")
    index = get_index(scene)
    matches: list[tuple[int, RegionMatch]] = []
    for entry in index.select(objects, exclude, include_hidden):
        relation = _region_relation(index, entry, region, mode)
        if relation is not None:
            matches.append((entry.order, RegionMatch(_ref(entry.obj), relation, mode != "bounds")))
    for obj, position in _point_objects(index, objects, exclude, include_hidden):
        if _points_in_region(region, position[None])[0]:
            matches.append((index.order[obj.id], RegionMatch(_ref(obj), "inside", True)))
    matches.sort(key=lambda m: m[0])
    return [m for _, m in matches]


# --------------------------------------------------------------------------- #
# Bounds and discovery
# --------------------------------------------------------------------------- #


def bounds(
    scene: Scene,
    obj: Any = None,
    *,
    space: str = "world",
    children: bool = False,
    include_hidden: bool = False,
) -> ObjectBounds | None:
    """Bounding volumes of an object's geometry, with ``children`` also its descendants'
    (``obj=None``: of the whole scene); ``None`` when there is no geometry.

    ``space="world"`` (default) or ``"local"`` (the object's own coordinates,
    before its transform). Hidden descendants (and, for the scene, hidden
    objects) count only with ``include_hidden``.
    """
    space = _choice(space, "space", ("world", "local"))
    children = _flag(children, "children")
    include_hidden = _flag(include_hidden, "include_hidden")
    index = get_index(scene)
    if obj is None:
        if space == "local":
            raise QueryError(
                "INVALID_ARGUMENT",
                "the scene as a whole has no local space",
                "space",
                hint="ask for the bounds of an object, or use space='world'",
            )
        target = None
        entries = index.select(None, None, include_hidden)
    else:
        target = index.resolve(obj, "object")
        members = {target.id} | ({o.id for o in target.iter_descendants()} if children else set())
        entries = [
            e
            for e in index.entries
            if e.obj.id in members and (e.obj is target or include_hidden or e.visible)
        ]
    if not entries:
        return None
    world = np.concatenate([e.points for e in entries])
    if target is not None:
        tw = index.worlds[target.id]
    if space == "local":
        own = len(entries) == 1 and entries[0].obj is target
        if own:
            local = entries[0].mesh.positions[entries[0].used].astype(np.float64)
        else:
            linear = tw[:3, :3]
            if abs(float(np.linalg.det(linear))) < 1e-300:
                raise QueryError(
                    "SINGULAR_TRANSFORM",
                    f"object {target.name!r} has a zero-scale world transform; its local space "
                    "cannot hold its children's geometry",
                    "object",
                )
            local = np.linalg.solve(linear, (world - tw[:3, 3]).T).T
        pts = local
    else:
        pts = world
    box = AABB.from_points(pts)
    center = np.asarray(box.center)
    radius = float(np.sqrt(np.einsum("ij,ij->i", pts - center, pts - center).max()))
    obb = None
    if space == "world" and target is not None:
        linear = tw[:3, :3]
        if len(entries) == 1 and entries[0].obj is target:
            local = entries[0].mesh.positions[entries[0].used].astype(np.float64)
        elif abs(float(np.linalg.det(linear))) >= 1e-300:
            local = np.linalg.solve(linear, (world - tw[:3, 3]).T).T
        else:
            local = None
        if local is not None:
            lo, hi = local.min(axis=0), local.max(axis=0)
            c = (lo + hi) * 0.5
            half = (hi - lo) * 0.5
            axes = tuple(_t3(linear[:, k] * half[k]) for k in range(3))
            obb = OrientedBox(_t3(linear @ c + tw[:3, 3]), axes)
    return ObjectBounds(
        None if target is None else _ref(target),
        space,
        box,
        obb,
        Sphere(_t3(center), radius),
        len(entries),
        int(sum(e.used.size for e in entries)),
        int(sum(e.mesh.triangle_count for e in entries)),
    )


def _strings(value: Any, what: str) -> list[str]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)) or not all(isinstance(t, str) for t in items):
        raise QueryError(
            "INVALID_ARGUMENT", f"must be a string or a list of strings, got {value!r}", what
        )
    return list(items)


def find_objects(
    scene: Scene,
    *,
    name: str | None = None,
    kind: Any = None,
    tags: Any = None,
    any_tags: Any = None,
    metadata: dict[str, Any] | None = None,
    metadata_keys: Any = None,
    parent: Any = None,
    ancestor: Any = None,
    visible: bool | None = None,
    material: Any = None,
    region: Any = None,
    region_mode: str = "intersects",
) -> list[SceneObject]:
    """Objects matching every given filter, in scene order (depth first).

    * ``name`` -- exact, or a case-sensitive glob pattern with ``*``, ``?`` or ``[...]``.
    * ``kind`` -- ``"mesh"``, ``"procedural"``, ``"camera"``, ``"light"``, ``"empty"``
      or ``"geometry"`` (mesh or procedural), or a list of them.
    * ``tags`` -- has all of these tags; ``any_tags`` -- has at least one.
    * ``metadata`` -- these entries with equal values; ``metadata_keys`` -- has these keys.
    * ``parent`` -- direct children of this object; ``ancestor`` -- its descendants.
    * ``visible`` -- ``True`` or ``False``.
    * ``material`` -- uses this material (its own, or for a face group).
    * ``region`` -- in this :class:`Box` or :class:`Sphere`, as
      :func:`query_region` with ``region_mode`` decides (hidden objects included).
    """
    index = get_index(scene)
    chosen = list(scene)
    if name is not None:
        if not isinstance(name, str):
            raise QueryError("INVALID_ARGUMENT", f"must be a string, got {name!r}", "name")
        if any(ch in name for ch in "*?["):
            chosen = [o for o in chosen if fnmatch.fnmatchcase(o.name, name)]
        else:
            chosen = [o for o in chosen if o.name == name]
    if kind is not None:
        kinds = set(_strings(kind, "kind"))
        for k in kinds:
            _choice(k, "kind", (*KINDS, "geometry"))
        if "geometry" in kinds:
            kinds |= {"mesh", "procedural"}
        chosen = [o for o in chosen if o.kind in kinds]
    if tags is not None:
        need = set(_strings(tags, "tags"))
        chosen = [o for o in chosen if need.issubset(o.tags)]
    if any_tags is not None:
        some = set(_strings(any_tags, "any_tags"))
        chosen = [o for o in chosen if some.intersection(o.tags)]
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise QueryError("INVALID_ARGUMENT", f"must be an object, got {metadata!r}", "metadata")
        chosen = [o for o in chosen if _metadata_match(o.metadata, metadata)]
    if metadata_keys is not None:
        keys = _strings(metadata_keys, "metadata_keys")
        chosen = [o for o in chosen if all(k in o.metadata for k in keys)]
    if parent is not None:
        p = index.resolve(parent, "parent")
        chosen = [o for o in chosen if o.parent is p]
    if ancestor is not None:
        a = index.resolve(ancestor, "ancestor")
        below = {o.id for o in a.iter_descendants()}
        chosen = [o for o in chosen if o.id in below]
    if visible is not None:
        visible = _flag(visible, "visible")
        chosen = [o for o in chosen if o.visible == visible]
    if material is not None:
        found = scene.materials.find_material(material)
        if found is None:
            raise QueryError(
                "MATERIAL_NOT_FOUND", f"no material with id or name {material!r}", "material"
            )
        chosen = [o for o in chosen if _uses_material(o, found.id)]
    if region is not None:
        ids = {
            m.object.id for m in query_region(scene, region, mode=region_mode, include_hidden=True)
        }
        chosen = [o for o in chosen if o.id in ids]
    return chosen


def _uses_material(obj: SceneObject, material_id: str) -> bool:
    own = obj.material
    if own is not None and own.id == material_id:
        return True
    return any(m.id == material_id for m in obj.group_materials.values())


def _metadata_match(have: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(key in have and _json_equal(have[key], value) for key, value in want.items())


def _json_equal(a: Any, b: Any) -> bool:
    """JSON equality: numbers by value (``3 == 3.0``) but booleans are not numbers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


class View:
    """A camera's view for an image size: its frame, extents and clipping range."""

    def __init__(self, scene: Scene, camera: Any, width: Any, height: Any) -> None:
        from ..errors import RenderError
        from ..rendering.camera import frame_from_matrix
        from ..rendering.scene_data import resolve_camera

        settings = scene.render_settings
        self.width = _integer(settings.width if width is None else width, "width", 1, 65536)
        self.height = _integer(settings.height if height is None else height, "height", 1, 65536)
        try:
            obj = resolve_camera(scene, camera)
            # The renderers' world matrices (the same arithmetic): camera rays match theirs.
            world = get_index(scene).worlds[obj.id]
            frame, _ = frame_from_matrix(world, f"objects[{obj.name}]")
            hx, hy = obj.camera.half_extents(self.width / self.height)
        except RenderError as exc:
            raise QueryError(exc.code, exc.message, "camera", hint=exc.hint) from None
        except AetherError as exc:
            message = exc.args[0] if exc.args else str(exc)
            raise QueryError("OBJECT_NOT_FOUND", str(message), "camera") from None
        self.obj = obj
        self.eye = np.asarray(frame.position, dtype=np.float64)
        self.forward = np.asarray(frame.forward, dtype=np.float64)
        self.right = np.asarray(frame.right, dtype=np.float64)
        self.up = np.asarray(frame.up, dtype=np.float64)
        self.hx, self.hy = hx, hy
        self.perspective = obj.camera.is_perspective
        self.near, self.far = obj.camera.near, obj.camera.far

    def rays(self, x: np.ndarray, y: np.ndarray):
        """Rays through image points (the path tracer's arithmetic): ``o, d, tmin, tmax``."""
        xn = x * (2.0 / self.width) - 1.0
        yn = 1.0 - y * (2.0 / self.height)
        sx = (xn * self.hx)[:, None]
        sy = (yn * self.hy)[:, None]
        n = xn.shape[0]
        if self.perspective:
            d = self.forward + sx * self.right + sy * self.up
            length = np.linalg.norm(d, axis=1, keepdims=True)
            o = np.broadcast_to(self.eye, (n, 3)).copy()
            return o, d / length, self.near * length[:, 0], self.far * length[:, 0]
        o = self.eye + sx * self.right + sy * self.up
        d = np.broadcast_to(self.forward, (n, 3)).copy()
        return o, d, np.full(n, self.near), np.full(n, self.far)

    def corners(self) -> np.ndarray:
        """The frustum's 8 corners: near plane (4), then far plane (4)."""
        out = []
        for depth in (self.near, self.far):
            scale = depth if self.perspective else 1.0
            center = self.eye + self.forward * depth
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                out.append(
                    center + sx * self.hx * scale * self.right + sy * self.hy * scale * self.up
                )
        return np.array(out)

    def planes(self) -> tuple[np.ndarray, np.ndarray]:
        """Outward unit normals ``(6, 3)`` and offsets: inside where ``n . p <= offset``."""
        c = self.corners()
        normals, offsets = [], []
        faces = ((0, 3, 2), (4, 5, 6), (0, 4, 7), (1, 2, 6), (0, 1, 5), (3, 7, 6))
        inside = c.mean(axis=0)
        for i, j, k in faces:
            n = np.cross(c[j] - c[i], c[k] - c[i])
            n = n / np.linalg.norm(n)
            if n @ (inside - c[i]) > 0.0:
                n = -n
            normals.append(n)
            offsets.append(float(n @ c[i]))
        return np.array(normals), np.array(offsets)

    def edges(self) -> np.ndarray:
        """Distinct edge directions of the frustum (for separating-axis tests)."""
        c = self.corners()
        return np.array([self.right, self.up, *(c[4:] - c[:4])])

    def project(self, pts: np.ndarray):
        rel = pts - self.eye
        xv = rel @ self.right
        yv = rel @ self.up
        depth = rel @ self.forward
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.perspective:
                xn = xv / (depth * self.hx)
                yn = yv / (depth * self.hy)
            else:
                xn = xv / self.hx
                yn = yv / self.hy
        x = (xn + 1.0) * (self.width / 2.0)
        y = (1.0 - yn) * (self.height / 2.0)
        if self.perspective:
            behind = depth <= 0.0
            x = np.where(behind, np.nan, x)
            y = np.where(behind, np.nan, y)
        inside = (
            (depth >= self.near)
            & (depth <= self.far)
            & (x >= 0.0)
            & (x < self.width)
            & (y >= 0.0)
            & (y < self.height)
        )
        return x, y, depth, inside


def camera_ray(
    scene: Scene, x: float, y: float, *, camera: Any = None, width: Any = None, height: Any = None
) -> CameraRay:
    """The ray a camera sees through image point ``(x, y)`` (pixels from the top-left corner;
    pixel ``(i, j)``'s center is ``(i + 0.5, j + 0.5)``). ``camera``: a camera object, id or
    name (default: the active camera); ``width`` x ``height``: the image size (default: the
    scene's render settings), which sets the aspect ratio."""
    view = View(scene, camera, width, height)
    o, d, tmin, tmax = view.rays(np.array([_number(x, "x")]), np.array([_number(y, "y")]))
    return CameraRay(_t3(o[0]), _t3(d[0]), float(tmin[0]), float(tmax[0]))


def pick(
    scene: Scene,
    x: float,
    y: float,
    *,
    camera: Any = None,
    width: Any = None,
    height: Any = None,
    objects: Any = None,
    exclude: Any = None,
    surfaces: str = "rendered",
    cull: str | None = None,
) -> SurfaceHit | None:
    """The surface a camera shows at image point ``(x, y)`` (see :func:`camera_ray`): the
    first hit between its near and far planes. ``surfaces="rendered"`` (default) skips what
    the renderers do not draw, including faces removed by ``cull`` (default: the scene's
    render settings); hidden objects never take part. ``distance`` is along the ray."""
    view = View(scene, camera, width, height)
    surfaces = _choice(surfaces, "surfaces", SURFACES)
    if cull is None:
        cull = scene.render_settings.cull if surfaces == "rendered" else "none"
    cull = _choice(cull, "cull", CULL_MODES)
    if surfaces == "geometry" and cull != "none":
        raise QueryError("INVALID_ARGUMENT", "culling applies to rendered surfaces only", "cull")
    o, d, tmin, tmax = view.rays(np.array([_number(x, "x")]), np.array([_number(y, "y")]))
    index = get_index(scene)
    entries = index.select(objects, exclude, False)
    hits, ents = _trace(index, entries, o, d, tmin, tmax, surfaces, cull, first=True)
    if not len(hits):
        return None
    return _hit(index, ents[hits.inst[0]], hits.prim[0], hits.b[0], hits.t[0], d[0])


def project(
    scene: Scene, points: Any, *, camera: Any = None, width: Any = None, height: Any = None
) -> ScreenPoint | list[ScreenPoint]:
    """Where world points appear in a camera's image (one point, or a list of them)."""
    view = View(scene, camera, width, height)
    arr = np.asarray(points, dtype=object)
    single = arr.shape == (3,)
    rows = [points] if single else list(points)
    pts = np.array([vec3(p, f"points[{k}]") for k, p in enumerate(rows)]).reshape(-1, 3)
    x, y, depth, inside = view.project(pts)
    out = [
        ScreenPoint(
            _t3(pts[k]),
            None if math.isnan(x[k]) else float(x[k]),
            None if math.isnan(y[k]) else float(y[k]),
            float(depth[k]),
            bool(inside[k]),
        )
        for k in range(pts.shape[0])
    ]
    return out[0] if single else out


def objects_in_view(
    scene: Scene,
    *,
    camera: Any = None,
    width: Any = None,
    height: Any = None,
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
    exact: bool = True,
) -> list[ViewMatch]:
    """Objects whose geometry is in a camera's view volume (between its near and far
    planes), in scene order: ``"inside"`` or ``"partial"``. Occlusion is not considered.

    ``exact`` (default) tests triangles against the frustum (separating axes);
    ``exact=False`` returns the objects whose world box meets it (candidates).
    """
    view = View(scene, camera, width, height)
    exact = _flag(exact, "exact")
    index = get_index(scene)
    corners = view.corners()
    normals, offsets = view.planes()
    edges = view.edges()
    flo, fhi = corners.min(axis=0), corners.max(axis=0)
    out: list[ViewMatch] = []
    for entry in index.select(objects, exclude, _flag(include_hidden, "include_hidden")):
        lo, hi = entry.box
        if np.any(lo > fhi) or np.any(hi < flo):
            continue
        box_vertices = geo.box_vertices(lo, hi)
        if not _convex_overlap(
            box_vertices, geo.BOX_NORMALS, geo.BOX_NORMALS, corners, normals, edges
        ):
            continue
        if not exact:
            inside = bool(
                np.all(box_vertices @ normals.T <= offsets + 1e-9 * (1 + np.abs(offsets)))
            )
            out.append(ViewMatch(_ref(entry.obj), "inside" if inside else "partial", False))
            continue
        pts = entry.points
        if np.all(pts @ normals.T <= offsets + 1e-9 * (1 + np.abs(offsets))):
            out.append(ViewMatch(_ref(entry.obj), "inside", True))
            continue
        _, tris = geo.bvh_box_pairs(entry.bvh, *entry.triangle_boxes, flo[None], fhi[None])
        if tris.size:
            a, b, c = entry.corners(tris)
            if geo.triangles_intersect_convex(a, b, c, corners, normals, edges).any():
                out.append(ViewMatch(_ref(entry.obj), "partial", True))
    return out


def _convex_overlap(va, na, ea, vb, nb, eb) -> bool:
    """Separating-axis test of two convex polyhedra (vertices, face normals, edge directions)."""
    axes = [*na, *nb]
    for x in ea:
        for y in eb:
            axis = np.cross(x, y)
            if np.linalg.norm(axis) > 1e-12 * max(1.0, np.linalg.norm(x) * np.linalg.norm(y)):
                axes.append(axis)
    for axis in axes:
        pa, pb = va @ axis, vb @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


# --------------------------------------------------------------------------- #
# Lines of sight and visibility
# --------------------------------------------------------------------------- #


def _endpoint(index: SpatialIndex, value: Any, what: str) -> tuple[np.ndarray, Any]:
    """A point, or an object's world position (and the object, to leave out as a blocker)."""
    from ..scene import SceneObject

    if isinstance(value, (SceneObject, str)):
        obj = index.resolve(value, what)
        return index.worlds[obj.id][:3, 3].copy(), obj
    return np.asarray(vec3(value, what)), None


def _segments(
    index: SpatialIndex,
    entries: list[Entry],
    start: np.ndarray,
    ends: np.ndarray,
    margin: float,
    surfaces: str,
):
    """First blocker of each segment ``start -> ends[k]`` (ignoring both ends' surroundings)."""
    rel = ends - start
    length = np.linalg.norm(rel, axis=1)
    usable = length > 0.0
    d = np.zeros_like(rel)
    d[usable] = rel[usable] / length[usable, None]
    scale = 1e-9 * (1.0 + np.maximum(np.abs(start).max(), np.abs(ends).max(axis=1)))
    eps = np.maximum(margin * length, scale)
    tmin, tmax = eps, length - eps
    live = usable & (tmax > tmin)
    rows = np.flatnonzero(live)
    o = np.broadcast_to(start, rel.shape)[rows].copy()
    hits, ents = _trace(index, entries, o, d[rows], tmin[rows], tmax[rows], surfaces, first=True)
    hits.ray = rows[hits.ray]
    return hits, ents, d, length


def line_of_sight(
    scene: Scene,
    start: Any,
    end: Any,
    *,
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
    surfaces: str = "geometry",
    margin: float = DEFAULT_MARGIN,
) -> LineOfSight:
    """Whether nothing lies on the segment between two points (or objects' positions).

    Surfaces within ``margin`` times the segment's length of either end do not
    block (so ends on a surface work); objects given as ends never block.
    """
    index = get_index(scene)
    p0, o0 = _endpoint(index, start, "start")
    p1, o1 = _endpoint(index, end, "end")
    margin = _number(margin, "margin", minimum=0.0)
    surfaces = _choice(surfaces, "surfaces", SURFACES)
    skip = [o for o in (o0, o1) if o is not None]
    if exclude is not None:
        skip += _as_list(exclude)
    entries = _entries(index, objects, skip or None, include_hidden, surfaces)
    hits, ents, d, length = _segments(index, entries, p0, p1[None], margin, surfaces)
    blocker = None
    if len(hits):
        blocker = _hit(index, ents[hits.inst[0]], hits.prim[0], hits.b[0], hits.t[0], d[0])
    return LineOfSight(_t3(p0), _t3(p1), blocker is None, float(length[0]), blocker)


def surface_samples(entry: Entry, count: int) -> np.ndarray:
    """``count`` points spread over the entry's world surface: deterministic, area-weighted
    (stratified over the triangles' cumulative area, an R2 sequence inside each)."""
    a, b, c = entry.corners()
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(area.sum())
    if not total > 0.0:
        return a[:0]
    cdf = np.cumsum(area) / total
    u = (np.arange(count) + 0.5) / count
    tri = np.minimum(np.searchsorted(cdf, u, side="right"), area.size - 1)
    g = 1.32471795724474602596
    s = (0.5 + np.arange(count) / g) % 1.0
    t = (0.5 + np.arange(count) / (g * g)) % 1.0
    flip = s + t > 1.0
    s, t = np.where(flip, 1.0 - s, s), np.where(flip, 1.0 - t, t)
    return a[tri] + s[:, None] * (b[tri] - a[tri]) + t[:, None] * (c[tri] - a[tri])


def visible_from(
    scene: Scene,
    target: Any,
    observer: Any,
    *,
    samples: int = 64,
    objects: Any = None,
    exclude: Any = None,
    include_hidden: bool = False,
    surfaces: str = "geometry",
    margin: float = DEFAULT_MARGIN,
    width: Any = None,
    height: Any = None,
) -> Visibility:
    """How much of ``target``'s surface ``observer`` (a point, or an object: its world
    position; a camera also limits the view to its frustum) sees.

    ``samples`` surface points (see :func:`surface_samples`) are joined to the
    observer by segments, as :func:`line_of_sight` tests them; ``objects`` and
    ``exclude`` select the possible blockers (the target always is one).
    """
    index = get_index(scene)
    entry = index.entry(target, "target")
    samples = _integer(samples, "samples", 1, MAX_SAMPLES)
    margin = _number(margin, "margin", minimum=0.0)
    surfaces = _choice(surfaces, "surfaces", SURFACES)
    eye, observer_obj = _endpoint(index, observer, "observer")
    skip = [] if observer_obj is None else [observer_obj]
    if exclude is not None:
        skip += _as_list(exclude)
    if surfaces == "rendered" and not entry.visible:
        raise QueryError(
            "INVALID_ARGUMENT",
            f"object {entry.obj.name!r} is hidden and does not render",
            "target",
            hint='use surfaces="geometry"',
        )
    blockers = _entries(index, objects, skip or None, include_hidden, surfaces)
    if entry not in blockers:
        blockers = sorted([*blockers, entry], key=lambda e: e.order)
    pts = surface_samples(entry, samples)
    in_view = np.ones(pts.shape[0], dtype=bool)
    if observer_obj is not None and observer_obj.camera is not None:
        in_view = View(scene, observer_obj, width, height).project(pts)[3]
    counts: dict[int, int] = {}
    visible = 0
    if pts.shape[0]:
        hits, ents, _, _ = _segments(index, blockers, eye, pts, margin, surfaces)
        blocked = np.zeros(pts.shape[0], dtype=bool)
        blocked[hits.ray] = True
        for k, inst in zip(hits.ray.tolist(), hits.inst.tolist(), strict=True):
            if in_view[k]:
                counts[ents[inst].order] = counts.get(ents[inst].order, 0) + 1
        visible = int(np.count_nonzero(~blocked & in_view))
    by_order = {e.order: e for e in index.entries}
    return Visibility(
        _ref(entry.obj),
        _t3(eye),
        int(pts.shape[0]),
        visible,
        tuple((_ref(by_order[k].obj), n) for k, n in sorted(counts.items())),
        int(np.count_nonzero(~in_view)),
    )


# --------------------------------------------------------------------------- #
# Relations
# --------------------------------------------------------------------------- #


def relation(
    scene: Scene,
    a: Any,
    b: Any,
    *,
    near_distance: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    viewer: Any = None,
    width: Any = None,
    height: Any = None,
) -> Relation:
    """How object ``a`` relates to object ``b`` in space (see :class:`~aether3d.spatial.Relation`
    for every predicate's definition)."""
    index = get_index(scene)
    ea = index.entry(a, "a")
    eb = index.entry(b, "b")
    if ea is eb:
        raise QueryError("INVALID_ARGUMENT", "a and b are the same object", "b")
    tol = _number(tolerance, "tolerance", minimum=0.0)
    near_limit = (
        None if near_distance is None else _number(near_distance, "near_distance", minimum=0.0)
    )
    dist, _, _, _ = _object_distance(ea, eb)
    intersecting = dist <= tol
    alo, ahi = ea.box
    blo, bhi = eb.box
    footprint = bool(np.all(alo[:2] <= bhi[:2] + tol) and np.all(blo[:2] <= ahi[:2] + tol))
    above = footprint and bool(alo[2] >= bhi[2] - tol)
    below = footprint and bool(ahi[2] <= blo[2] + tol)
    inside = contains_ = None
    if index.topology(eb)["solid"]:
        inside = (not intersecting) and _all_inside(index, ea, eb)
    if index.topology(ea)["solid"]:
        contains_ = (not intersecting) and _all_inside(index, eb, ea)
    forward = -eb.world[:3, 1]
    length = float(np.linalg.norm(forward))
    in_front = behind = False
    if length > 1e-12:
        forward = forward / length
        pa = ea.points @ forward
        pb = eb.points @ forward
        in_front = bool(pa.min() >= pb.max() - tol)
        behind = bool(pa.max() <= pb.min() + tol)
    order = None
    if viewer is not None:
        order = _viewer_order(scene, index, ea, eb, viewer, tol, width, height)
    return Relation(
        _ref(ea.obj),
        _ref(eb.obj),
        dist,
        intersecting,
        None if near_limit is None else dist <= near_limit,
        above,
        below,
        inside,
        contains_,
        in_front,
        behind,
        order,
        tol,
    )


def _viewer_order(scene, index, ea, eb, viewer, tol, width, height) -> str:
    """Which object is nearer to a viewer: by view depth for a camera, by distance for a point."""
    from ..scene import SceneObject

    camera = None
    if isinstance(viewer, (SceneObject, str)):
        obj = index.resolve(viewer, "viewer")
        if obj.camera is not None:
            camera = View(scene, obj, width, height)
        eye = index.worlds[obj.id][:3, 3].copy()
    else:
        eye = np.asarray(vec3(viewer, "viewer"))
    if camera is not None:
        da = (ea.points - camera.eye) @ camera.forward
        db = (eb.points - camera.eye) @ camera.forward
        a_range = (float(da.min()), float(da.max()))
        b_range = (float(db.min()), float(db.max()))
    else:
        na = _nearest(ea, eye)
        nb = _nearest(eb, eye)
        a_range = (math.sqrt(na[0]), float(np.linalg.norm(ea.points - eye, axis=1).max()))
        b_range = (math.sqrt(nb[0]), float(np.linalg.norm(eb.points - eye, axis=1).max()))
    if a_range[1] <= b_range[0] + tol:
        return "a_nearer"
    if b_range[1] <= a_range[0] + tol:
        return "b_nearer"
    return "overlapping"


# --------------------------------------------------------------------------- #
# A compact description of the scene
# --------------------------------------------------------------------------- #

DESCRIBE_FORMAT = "aether3d.spatial-scene"
DESCRIBE_VERSION = 1


def describe(scene: Scene, *, include_hidden: bool = True) -> dict[str, Any]:
    """A compact JSON summary of the scene for tools and agents: every object's id, name,
    kind, parent, visibility, tags, world position and world box; geometry counts; and
    whether its mesh is a closed solid. Conventions are spelled out in ``frame``."""
    include_hidden = _flag(include_hidden, "include_hidden")
    index = get_index(scene)
    worlds = index.worlds
    objects = []
    for obj in scene:
        if not include_hidden and not obj.visible:
            continue
        record: dict[str, Any] = {
            "id": obj.id,
            "name": obj.name,
            "kind": obj.kind,
            "parent": None if obj.parent is None else obj.parent.id,
        }
        if not obj.visible:
            record["visible"] = False
        if obj.tags:
            record["tags"] = list(obj.tags)
        metadata = obj.metadata
        if metadata:
            record["metadata"] = metadata
        record["position"] = [float(v) for v in worlds[obj.id][:3, 3]]
        if obj.id in index.failed:
            record["error"] = index.failed[obj.id].to_dict()
        entry = index.by_id.get(obj.id)
        if entry is not None:
            lo, hi = entry.box
            record["bounds"] = {"min": lo.tolist(), "max": hi.tolist()}
            record["triangles"] = entry.mesh.triangle_count
            record["closed"] = bool(index.topology(entry)["solid"])
            material = obj.material
            if material is not None:
                record["material"] = material.name
        objects.append(record)
    entries = [
        e for e in index.entries if include_hidden or e.visible
    ]  # (objects that failed to evaluate are reported above, not bounded)
    if entries:
        pts = np.concatenate([e.points for e in entries])
        whole = {"min": pts.min(axis=0).tolist(), "max": pts.max(axis=0).tolist()}
    else:
        whole = None
    active = scene.active_camera
    return {
        "format": DESCRIBE_FORMAT,
        "version": DESCRIBE_VERSION,
        "scene": scene.name,
        "revision": scene.revision,
        "frame": {
            "handedness": "right",
            "up": [0.0, 0.0, 1.0],
            "front": [0.0, -1.0, 0.0],
            "units": "scene units (meters by convention)",
        },
        "active_camera": None if active is None else active.id,
        "bounds": whole,
        "objects": objects,
    }
