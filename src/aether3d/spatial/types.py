"""Regions and result types of the spatial queries.

Every result is a frozen dataclass with a ``to_dict()`` that gives the JSON form the
command line prints. Positions, directions and distances are in **world space**
(Aether3D's right-handed, +Z-up frame, in scene units) unless a field says
otherwise; objects are referred to by id and name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import QueryError

Vec3 = tuple[float, float, float]


def vec3(value: Any, what: str) -> Vec3:
    """A finite 3-vector, or ``QueryError("INVALID_ARGUMENT")``."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        arr = None
    if arr is None or arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise QueryError("INVALID_ARGUMENT", f"must be three finite numbers, got {value!r}", what)
    x, y, z = arr.tolist()
    return (x, y, z)


def _list(v: Any) -> list[float]:
    return [float(x) for x in v]


@dataclass(frozen=True)
class ObjectRef:
    """An object of the scene: its stable id and its (changeable) name."""

    id: str
    name: str

    @classmethod
    def of(cls, obj: Any) -> ObjectRef:
        return cls(obj.id, obj.name)

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name}


# --------------------------------------------------------------------------- #
# Bounding volumes and regions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AABB:
    """An axis-aligned box ``[min, max]`` (closed). Also a query region (``Box``)."""

    min: Vec3
    max: Vec3

    def __post_init__(self) -> None:
        lo = vec3(self.min, "box min")
        hi = vec3(self.max, "box max")
        if any(a > b for a, b in zip(lo, hi, strict=True)):
            raise QueryError("INVALID_ARGUMENT", f"box min {lo} is above its max {hi}")
        object.__setattr__(self, "min", lo)
        object.__setattr__(self, "max", hi)

    @classmethod
    def from_points(cls, points: np.ndarray) -> AABB:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        return cls(tuple(lo.tolist()), tuple(hi.tolist()))

    @property
    def center(self) -> Vec3:
        x, y, z = ((a + b) * 0.5 for a, b in zip(self.min, self.max, strict=True))
        return (x, y, z)

    @property
    def size(self) -> Vec3:
        x, y, z = (b - a for a, b in zip(self.min, self.max, strict=True))
        return (x, y, z)

    @property
    def volume(self) -> float:
        x, y, z = self.size
        return x * y * z

    def corners(self) -> np.ndarray:
        """The eight corners, ``(8, 3)``."""
        lo, hi = self.min, self.max
        return np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )

    def contains(self, point: Any, tolerance: float = 0.0) -> bool:
        p = vec3(point, "point")
        return all(
            a - tolerance <= v <= b + tolerance
            for a, v, b in zip(self.min, p, self.max, strict=True)
        )

    def overlaps(self, other: AABB, tolerance: float = 0.0) -> bool:
        return all(
            a1 <= b2 + tolerance and a2 <= b1 + tolerance
            for a1, b1, a2, b2 in zip(self.min, self.max, other.min, other.max, strict=True)
        )

    def union(self, other: AABB) -> AABB:
        lo = tuple(min(a, b) for a, b in zip(self.min, other.min, strict=True))
        hi = tuple(max(a, b) for a, b in zip(self.max, other.max, strict=True))
        return AABB(lo, hi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": _list(self.min),
            "max": _list(self.max),
            "center": _list(self.center),
            "size": _list(self.size),
        }


Box = AABB
"""A box region for :meth:`Scene.query_region` and ``find_objects(region=...)``."""


@dataclass(frozen=True)
class Sphere:
    """A sphere (closed). A query region, and the bounding sphere of geometry."""

    center: Vec3
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", vec3(self.center, "sphere center"))
        r = self.radius
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0:
            raise QueryError(
                "INVALID_ARGUMENT", f"sphere radius must be a finite number >= 0, got {r!r}"
            )
        object.__setattr__(self, "radius", float(r))

    def contains(self, point: Any, tolerance: float = 0.0) -> bool:
        p = np.asarray(vec3(point, "point"))
        return float(np.linalg.norm(p - self.center)) <= self.radius + tolerance

    def to_dict(self) -> dict[str, Any]:
        return {"center": _list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class OrientedBox:
    """An object's local bounding box carried into world space by its world matrix.

    ``center`` plus the three ``half_axes`` (the box's local x, y and z half edges,
    transformed) give every corner exactly: ``center +- h0 +- h1 +- h2``. Under a
    parent's non-uniform scale combined with rotation the box shears into a
    parallelepiped; ``orthogonal`` says whether the half axes are perpendicular.
    """

    center: Vec3
    half_axes: tuple[Vec3, Vec3, Vec3]

    @property
    def half_extents(self) -> Vec3:
        x, y, z = (float(np.linalg.norm(a)) for a in self.half_axes)
        return (x, y, z)

    @property
    def orthogonal(self) -> bool:
        h = np.asarray(self.half_axes)
        scale = max(float(np.abs(h).max()), 1e-300) ** 2
        return all(abs(float(h[i] @ h[j])) <= 1e-9 * scale for i, j in ((0, 1), (0, 2), (1, 2)))

    def corners(self) -> np.ndarray:
        h = np.asarray(self.half_axes)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return np.asarray(self.center) + signs @ h

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": _list(self.center),
            "half_axes": [_list(a) for a in self.half_axes],
            "half_extents": _list(self.half_extents),
            "orthogonal": self.orthogonal,
        }


@dataclass(frozen=True)
class ObjectBounds:
    """Bounding volumes of an object's geometry (or of it and its descendants).

    * ``aabb`` -- the tight axis-aligned box of the geometry's vertices in ``space``
      (``"world"``, or ``"local"``: the object's own coordinates, before its
      transform).
    * ``obb`` -- (world space, one object) the local box carried into world space.
    * ``sphere`` -- a sphere enclosing every vertex: centered on the box, with the
      largest vertex distance as radius. It encloses, but is not the smallest one.
    * ``objects`` -- how many objects with geometry contributed (``children=True``
      aggregates descendants), ``vertices`` and ``triangles`` their totals.
    """

    object: ObjectRef | None  # None: the whole scene
    space: str
    aabb: AABB
    obb: OrientedBox | None
    sphere: Sphere
    objects: int
    vertices: int
    triangles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": None if self.object is None else self.object.to_dict(),
            "space": self.space,
            "aabb": self.aabb.to_dict(),
            "obb": None if self.obb is None else self.obb.to_dict(),
            "sphere": self.sphere.to_dict(),
            "objects": self.objects,
            "vertices": self.vertices,
            "triangles": self.triangles,
        }


# --------------------------------------------------------------------------- #
# Surface results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SurfaceHit:
    """A point on an object's surface: a ray hit or a closest point.

    * ``distance`` -- ray hits: along the (normalized) ray from its origin;
      closest points: from the query point.
    * ``normal`` -- the triangle's geometric normal (unit, world space) on its
      front side, the side its corners wind counterclockwise around (mirrored
      transforms are accounted for). ``front_face`` says whether a ray arrived from
      that side (always ``None`` for closest points).
    * ``shading_normal`` -- the mesh's vertex normals interpolated and carried into
      world space (unit), or the geometric normal if the mesh has none.
    * ``triangle`` -- the triangle's index in the object's mesh (the evaluated mesh
      of a procedural object); ``face_group`` its face group (0 without groups);
      ``barycentric`` the weights of its corners 0, 1 and 2; ``uv`` the
      interpolated texture coordinates (``None`` without UVs).
    * ``material`` -- the material its face group renders with (``None``: the
      object's own color).
    * ``instance_of`` -- objects whose geometry is one shared mesh (copies made by
      ``duplicate``, objects of one procedural node) are instances of it: the
      first of them in scene order (the object itself when its mesh is its own);
      ``instance_count`` -- how many objects share the mesh.
    """

    object: ObjectRef
    position: Vec3
    distance: float
    normal: Vec3
    shading_normal: Vec3
    front_face: bool | None
    triangle: int
    face_group: int
    barycentric: Vec3
    uv: tuple[float, float] | None
    material: ObjectRef | None
    instance_of: ObjectRef
    instance_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object.to_dict(),
            "position": _list(self.position),
            "distance": self.distance,
            "normal": _list(self.normal),
            "shading_normal": _list(self.shading_normal),
            "front_face": self.front_face,
            "triangle": self.triangle,
            "face_group": self.face_group,
            "barycentric": _list(self.barycentric),
            "uv": None if self.uv is None else _list(self.uv),
            "material": None if self.material is None else self.material.to_dict(),
            "instance_of": self.instance_of.to_dict(),
            "instance_count": self.instance_count,
        }


RayHit = SurfaceHit
SurfacePoint = SurfaceHit


@dataclass(frozen=True)
class Containment:
    """Whether a point is inside an object's solid.

    ``state`` is ``"inside"``, ``"outside"`` or ``"surface"`` (within ``tolerance``
    of the surface) for a closed mesh -- watertight, edge-manifold and consistently
    oriented -- decided by its winding number. For any other mesh the state is
    ``"undefined"``: an open or non-manifold surface bounds no solid. ``winding``
    is the generalized winding number either way (fractional for open meshes);
    ``reason`` says why the state is undefined.
    """

    object: ObjectRef
    point: Vec3
    state: str
    winding: float
    distance: float
    closed: bool
    reason: str | None

    @property
    def inside(self) -> bool | None:
        return {"inside": True, "outside": False}.get(self.state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object.to_dict(),
            "point": _list(self.point),
            "state": self.state,
            "winding": self.winding,
            "distance": self.distance,
            "closed": self.closed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ObjectDistance:
    """The shortest distance between two objects' surfaces and where it is reached
    (``0`` and ``intersecting`` if they touch or cross)."""

    a: ObjectRef
    b: ObjectRef
    distance: float
    point_a: Vec3
    point_b: Vec3
    intersecting: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "distance": self.distance,
            "point_a": _list(self.point_a),
            "point_b": _list(self.point_b),
            "intersecting": self.intersecting,
        }


@dataclass(frozen=True)
class Overlap:
    """Two objects whose geometry overlaps, or whose bounds do.

    * ``bounds_overlap`` -- their world boxes overlap: a broad-phase candidate only.
    * ``intersecting`` -- their surfaces share a point (exact triangle tests;
      ``None`` when only bounds were compared).
    * ``contained`` -- ``"a_in_b"`` or ``"b_in_a"`` when one is entirely inside the
      other's closed solid without their surfaces meeting (``None`` otherwise).
    * ``exact`` -- whether the triangles were tested.
    """

    a: ObjectRef
    b: ObjectRef
    bounds_overlap: bool
    intersecting: bool | None
    contained: str | None
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "bounds_overlap": self.bounds_overlap,
            "intersecting": self.intersecting,
            "contained": self.contained,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class RegionMatch:
    """An object found in a region.

    ``relation``: ``"inside"`` (every vertex in the region), ``"intersects"`` (the
    surface meets the region), ``"contains"`` (the region lies inside the object's
    closed solid without touching its surface) or ``"bounds"`` (only the world box
    meets the region: a candidate, ``exact`` is false).
    """

    object: ObjectRef
    relation: str
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {"object": self.object.to_dict(), "relation": self.relation, "exact": self.exact}


@dataclass(frozen=True)
class CameraRay:
    """The ray through a point of a camera's image (what the path tracer traces).

    ``origin`` and the unit ``direction`` in world space; the camera sees
    surfaces between ``min_distance`` and ``max_distance`` along it (its near and
    far planes, which are view depths, converted to distances along this ray).
    """

    origin: Vec3
    direction: Vec3
    min_distance: float
    max_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": _list(self.origin),
            "direction": _list(self.direction),
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
        }


@dataclass(frozen=True)
class ScreenPoint:
    """A world point projected into an image.

    ``x``, ``y`` are pixels from the image's top-left corner: pixel ``(i, j)``
    covers ``[i, i + 1) x [j, j + 1)``, its center is ``(i + 0.5, j + 0.5)``
    (``None`` for points a perspective camera cannot project: at or behind its
    eye). ``depth`` is the distance in front of the camera along its view
    direction (negative behind it). ``in_view``: inside the image and between
    the near and far planes.
    """

    point: Vec3
    x: float | None
    y: float | None
    depth: float
    in_view: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": _list(self.point),
            "x": self.x,
            "y": self.y,
            "depth": self.depth,
            "in_view": self.in_view,
        }


@dataclass(frozen=True)
class ViewMatch:
    """An object's place relative to a camera's view volume (frustum).

    ``visibility``: ``"inside"`` (all geometry in the frustum), ``"partial"`` (it
    crosses the frustum's boundary) or ``"outside"``. ``exact``: triangles were
    tested (``False``: only the world box, a conservative candidate). Occlusion is
    not considered (see ``visible_from``).
    """

    object: ObjectRef
    visibility: str
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object.to_dict(),
            "visibility": self.visibility,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class LineOfSight:
    """Whether the open segment between two points is free of geometry.

    ``blocker`` is the first surface on the way (``None`` when ``clear``);
    ``distance`` the length of the segment.
    """

    start: Vec3
    end: Vec3
    clear: bool
    distance: float
    blocker: SurfaceHit | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _list(self.start),
            "end": _list(self.end),
            "clear": self.clear,
            "distance": self.distance,
            "blocker": None if self.blocker is None else self.blocker.to_dict(),
        }


@dataclass(frozen=True)
class Visibility:
    """How much of an object's surface an observer sees.

    ``samples`` points spread over the object's surface (deterministically,
    proportionally to area) are joined to the observer; ``visible`` counts those
    whose segment meets no surface, ``fraction`` is ``visible / samples``, and
    ``blockers`` counts the objects that hid the others (the target itself hides
    its far side). With a camera as observer, samples outside its view volume
    are not visible either (``outside_view``).
    """

    target: ObjectRef
    observer: Vec3
    samples: int
    visible: int
    blockers: tuple[tuple[ObjectRef, int], ...]
    outside_view: int = 0

    @property
    def fraction(self) -> float:
        return self.visible / self.samples if self.samples else 0.0

    @property
    def is_visible(self) -> bool:
        return self.visible > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "observer": _list(self.observer),
            "samples": self.samples,
            "visible": self.visible,
            "fraction": self.fraction,
            "outside_view": self.outside_view,
            "blockers": [{"object": o.to_dict(), "samples": n} for o, n in self.blockers],
        }


@dataclass(frozen=True)
class Relation:
    """Spatial relationships of object ``a`` to object ``b`` (see docs/SPATIAL.md).

    Booleans are exact predicates on the objects' world geometry, with
    ``tolerance`` in world units:

    * ``intersecting`` -- the surfaces share a point; ``distance`` is the shortest
      distance between the surfaces (0 when intersecting).
    * ``near`` -- ``distance <= near_distance`` (``None`` without a threshold).
    * ``above`` -- ``a``'s lowest vertex is at or above ``b``'s highest one and their
      footprints (world boxes projected on the XY plane) overlap; ``below`` the
      reverse.
    * ``inside`` -- every vertex of ``a`` is inside ``b``'s closed solid and the
      surfaces do not intersect (``None`` if ``b`` is not closed); ``contains`` the
      reverse.
    * ``in_front_of`` -- ``a`` lies entirely beyond ``b``'s front, the plane at
      ``b``'s foremost vertex across ``b``'s world forward direction (its local
      -Y, Aether3D's front); ``behind`` the reverse, beyond ``b``'s back.
    * ``viewer`` -- with a viewer: ``"a_nearer"``, ``"b_nearer"`` or
      ``"overlapping"``, comparing the objects' depth ranges along the viewer's
      view direction (a camera's forward, or from a point toward the objects).
    """

    a: ObjectRef
    b: ObjectRef
    distance: float
    intersecting: bool
    near: bool | None
    above: bool
    below: bool
    inside: bool | None
    contains: bool | None
    in_front_of: bool
    behind: bool
    viewer: str | None
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "distance": self.distance,
            "intersecting": self.intersecting,
            "near": self.near,
            "above": self.above,
            "below": self.below,
            "inside": self.inside,
            "contains": self.contains,
            "in_front_of": self.in_front_of,
            "behind": self.behind,
            "viewer": self.viewer,
            "tolerance": self.tolerance,
        }
