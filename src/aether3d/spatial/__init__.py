"""Scene intelligence and spatial queries (0.10.0).

Read-only questions about where things are, answered exactly from the scene's
triangles in world space: bounding volumes, object discovery, ray casts, nearest
surface points, point containment, object distances and overlaps, region
queries, camera frusta, picking and projection, lines of sight, visibility and
spatial relations. Every query is also a :class:`~aether3d.Scene` method and a
JSON request (:func:`run_query`; command line: ``aether3d query``).

See ``docs/SPATIAL.md`` for conventions, tolerances and limitations.
"""

from .index import SpatialIndex, get_index
from .queries import (
    bounds,
    camera_ray,
    closest_point,
    contains_point,
    describe,
    distance,
    find_objects,
    line_of_sight,
    nearest_objects,
    objects_in_view,
    overlaps,
    pick,
    project,
    query_region,
    raycast,
    raycast_all,
    relation,
    surface_samples,
    visible_from,
)
from .request import QUERIES, run_query
from .types import (
    AABB,
    Box,
    CameraRay,
    Containment,
    LineOfSight,
    ObjectBounds,
    ObjectDistance,
    ObjectRef,
    OrientedBox,
    Overlap,
    RayHit,
    RegionMatch,
    Relation,
    ScreenPoint,
    Sphere,
    SurfaceHit,
    SurfacePoint,
    ViewMatch,
    Visibility,
)

__all__ = [
    "AABB",
    "Box",
    "CameraRay",
    "Containment",
    "LineOfSight",
    "ObjectBounds",
    "ObjectDistance",
    "ObjectRef",
    "OrientedBox",
    "Overlap",
    "QUERIES",
    "RayHit",
    "RegionMatch",
    "Relation",
    "ScreenPoint",
    "SpatialIndex",
    "Sphere",
    "SurfaceHit",
    "SurfacePoint",
    "ViewMatch",
    "Visibility",
    "bounds",
    "camera_ray",
    "closest_point",
    "contains_point",
    "describe",
    "distance",
    "find_objects",
    "get_index",
    "line_of_sight",
    "nearest_objects",
    "objects_in_view",
    "overlaps",
    "pick",
    "project",
    "query_region",
    "raycast",
    "raycast_all",
    "relation",
    "run_query",
    "surface_samples",
    "visible_from",
]
