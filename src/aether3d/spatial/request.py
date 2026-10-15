"""Spatial queries as JSON requests: what the command line and scene scripts send.

A request is a JSON object naming its query and giving its arguments::

    {"query": "raycast", "origin": [0, 0, 10], "direction": [0, 0, -1]}

:func:`run_query` answers it with ``{"query": name, "result": ...}``, where the
result is the JSON form (``to_dict()``) of what the Python function returns.
Arguments keep their Python names; objects are ids or names (or lists of them),
points are ``[x, y, z]``, regions are ``{"box": {"min": [...], "max": [...]}}``
or ``{"sphere": {"center": [...], "radius": r}}``, and a ``max_distance`` of
``null`` means unlimited.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..errors import QueryError
from . import queries
from .types import AABB, Sphere

if TYPE_CHECKING:
    from ..scene import Scene


@dataclass(frozen=True)
class Query:
    """A query's function and its JSON arguments: ``required`` (passed in this order after
    the scene), ``optional`` positional ones and ``keywords``."""

    function: Callable[..., Any]
    required: tuple[str, ...]
    keywords: tuple[str, ...]
    optional: tuple[str, ...] = ()
    summary: str = ""

    @property
    def arguments(self) -> tuple[str, ...]:
        return (*self.required, *self.optional, *self.keywords)


_SELECT = ("objects", "exclude", "include_hidden")
_CAMERA = ("camera", "width", "height")

QUERIES: dict[str, Query] = {
    "describe": Query(
        queries.describe, (), ("include_hidden",), summary="compact summary of every object"
    ),
    "bounds": Query(
        queries.bounds,
        (),
        ("space", "children", "include_hidden"),
        ("object",),
        "bounding box, oriented box and sphere of an object (or the scene)",
    ),
    "find": Query(
        queries.find_objects,
        (),
        (
            "name",
            "kind",
            "tags",
            "any_tags",
            "metadata",
            "metadata_keys",
            "parent",
            "ancestor",
            "visible",
            "material",
            "region",
            "region_mode",
        ),
        summary="objects matching filters",
    ),
    "raycast": Query(
        queries.raycast,
        ("origin", "direction"),
        ("max_distance", "min_distance", *_SELECT, "surfaces"),
        summary="first surface along a ray",
    ),
    "raycast_all": Query(
        queries.raycast_all,
        ("origin", "direction"),
        ("max_distance", "min_distance", *_SELECT, "surfaces", "limit"),
        summary="every surface along a ray",
    ),
    "closest_point": Query(
        queries.closest_point,
        ("point",),
        ("max_distance", *_SELECT),
        summary="nearest surface point",
    ),
    "nearest": Query(
        queries.nearest_objects,
        ("point",),
        ("count", "max_distance", *_SELECT),
        summary="objects by surface distance",
    ),
    "contains": Query(
        queries.contains_point,
        ("object", "point"),
        ("tolerance",),
        summary="whether a point is inside an object's solid",
    ),
    "distance": Query(
        queries.distance, ("a", "b"), (), summary="shortest distance between two objects"
    ),
    "overlaps": Query(
        queries.overlaps,
        (),
        ("objects", "against", "exclude", "include_hidden", "exact", "tolerance"),
        summary="pairs of overlapping objects",
    ),
    "region": Query(
        queries.query_region,
        ("region",),
        ("mode", *_SELECT),
        summary="objects in a box or sphere",
    ),
    "camera_ray": Query(
        queries.camera_ray, ("x", "y"), _CAMERA, summary="the ray through an image point"
    ),
    "pick": Query(
        queries.pick,
        ("x", "y"),
        (*_CAMERA, "objects", "exclude", "surfaces", "cull"),
        summary="the surface shown at an image point",
    ),
    "project": Query(
        queries.project, ("points",), _CAMERA, summary="image positions of world points"
    ),
    "in_view": Query(
        queries.objects_in_view,
        (),
        (*_CAMERA, *_SELECT, "exact"),
        summary="objects in a camera's view volume",
    ),
    "line_of_sight": Query(
        queries.line_of_sight,
        ("start", "end"),
        (*_SELECT, "surfaces", "margin"),
        summary="whether a segment is unobstructed",
    ),
    "visible_from": Query(
        queries.visible_from,
        ("target", "observer"),
        ("samples", *_SELECT, "surfaces", "margin", "width", "height"),
        summary="how much of an object an observer sees",
    ),
    "relation": Query(
        queries.relation,
        ("a", "b"),
        ("near_distance", "tolerance", "viewer", "width", "height"),
        summary="spatial relations between two objects",
    ),
}
"""Every query by its request name."""

OBJECT_ARGUMENTS = frozenset(
    {"object", "a", "b", "target", "camera", "parent", "ancestor", "material"}
)
OBJECT_LISTS = frozenset({"objects", "exclude", "against"})
POINT_OR_OBJECT = frozenset({"start", "end", "observer", "viewer"})


def region_from_json(value: Any, location: str = "region") -> AABB | Sphere:
    """A region from its JSON form."""
    if isinstance(value, (AABB, Sphere)):
        return value
    if isinstance(value, dict) and len(value) == 1:
        if isinstance(value.get("box"), dict) and set(value["box"]) == {"min", "max"}:
            return AABB(value["box"]["min"], value["box"]["max"])
        if isinstance(value.get("sphere"), dict) and set(value["sphere"]) == {"center", "radius"}:
            return Sphere(value["sphere"]["center"], value["sphere"]["radius"])
    raise QueryError(
        "INVALID_REQUEST",
        f"a region is {{'box': {{'min': [x, y, z], 'max': [x, y, z]}}}} or "
        f"{{'sphere': {{'center': [x, y, z], 'radius': r}}}}, got {value!r}",
        location,
    )


def _argument(name: str, value: Any) -> Any:
    if name == "region":
        return region_from_json(value)
    if name == "max_distance" and value is None:
        return math.inf
    return value


def to_json(value: Any) -> Any:
    """The JSON form of a query result."""
    from ..scene import SceneObject

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, SceneObject):
        return {"id": value.id, "name": value.name, "kind": value.kind}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, dict):
        return value
    return value.to_dict()


def run_query(scene: Scene, request: Any) -> dict[str, Any]:
    """Answer one request (see the module docstring); ``QueryError`` if it is malformed.

    The scene is only read: no history step, no revision change.
    """
    if not isinstance(request, dict):
        raise QueryError("INVALID_REQUEST", f"a request is a JSON object, got {request!r}")
    name = request.get("query")
    query = QUERIES.get(name) if isinstance(name, str) else None
    if query is None:
        raise QueryError(
            "INVALID_REQUEST",
            f"unknown query {name!r}; queries: {', '.join(QUERIES)}",
            "query",
        )
    unknown = sorted(set(request) - {"query", *query.arguments})
    if unknown:
        raise QueryError(
            "INVALID_REQUEST",
            f"unknown argument(s) {', '.join(unknown)} for {name!r}; it takes "
            + (", ".join(query.arguments) or "no arguments"),
            unknown[0],
        )
    missing = [k for k in query.required if k not in request]
    if missing:
        raise QueryError("INVALID_REQUEST", f"{name!r} needs {', '.join(missing)}", missing[0])
    args = [_argument(k, request[k]) for k in query.required]
    args += [_argument(k, request[k]) for k in query.optional if k in request]
    kwargs = {k: _argument(k, request[k]) for k in query.keywords if k in request}
    result = query.function(scene, *args, **kwargs)
    return {"query": name, "result": to_json(result)}


_UNPARSED = object()


def parse_value(name: str, text: str) -> Any:
    """A command-line value: JSON if it parses, else comma-separated numbers as a list,
    else the text itself. Arguments naming objects keep plain text unless it is JSON text or
    a JSON list (an object may be named ``1`` or ``true``)."""
    try:
        value = json.loads(text)
    except ValueError:
        value = _UNPARSED
    if value is not _UNPARSED:
        names_objects = name in OBJECT_ARGUMENTS | OBJECT_LISTS | POINT_OR_OBJECT
        return text if names_objects and not isinstance(value, (str, list)) else value
    if "," in text and name not in OBJECT_ARGUMENTS:
        parts = [p.strip() for p in text.split(",")]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return parts if name in OBJECT_LISTS else text
    return text


def request_from_arguments(name: str, pairs: Sequence[str]) -> dict[str, Any]:
    """A request from ``KEY=VALUE`` command-line arguments (see :func:`parse_value`)."""
    request: dict[str, Any] = {"query": name}
    for pair in pairs:
        key, sep, text = pair.partition("=")
        key = key.strip().replace("-", "_")
        if not sep or not key:
            raise QueryError("INVALID_REQUEST", f"expected KEY=VALUE, got {pair!r}", pair)
        if key in request:
            raise QueryError("INVALID_REQUEST", f"{key} is given twice", key)
        request[key] = parse_value(key, text)
    return request
