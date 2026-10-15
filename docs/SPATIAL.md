# Scene intelligence and spatial queries

Aether3D answers questions about where things are in a scene: what a ray hits,
what is nearest to a point, whether a point is inside an object, which objects
touch, what is in a region or in a camera's view, what a click on an image
picks, whether one point sees another, and how two objects relate. Every answer
is computed from the triangles themselves, in world space, and changes nothing.

```python
from aether3d import Scene
from aether3d.spatial import Box

scene = Scene("Yard")
scene.add_plane(name="Ground", size=20)
crate = scene.add_cube(name="Crate", size=1, position=(2, 0, 0.5))
crate.tags = ["prop", "movable"]
scene.add_sphere(name="Ball", radius=0.5, position=(-2, 0, 0.5))
scene.add_camera("Cam", position=(0, -8, 3), look_at=(0, 0, 0.5))

hit = scene.raycast((2, 0, 10), (0, 0, -1))  # straight down onto the crate
print(hit.object.name, hit.position, hit.normal)  # Crate (2.0, 0.0, 1.0) (0.0, 0.0, 1.0)
near = scene.nearest_objects((0, 0, 0.5), count=2)
print([(h.object.name, round(h.distance, 3)) for h in near])  # [('Ground', 0.5), ('Crate', 1.5)]
# (the ball is 1.5 away too: equal distances go by scene order)
print(scene.contains_point("Crate", (2, 0, 0.5)).state)  # inside
in_box = scene.query_region(Box((1, -1, 0), (3, 1, 2)))
print([m.object.name for m in in_box])  # ['Ground', 'Crate']
print(scene.pick(640, 360).object.name)  # Ground: what the camera shows at the image's center
print(scene.relation("Crate", "Ground").above)  # True
print([o.name for o in scene.find_objects(tags="prop")])  # ['Crate']
```

Contents:
- [Conventions](#conventions)
- [Tags and metadata](#tags-and-metadata)
- [Bounds](#bounds)
- [Finding objects](#finding-objects)
- [Rays](#rays)
- [Nearest points and distances](#nearest-points-and-distances)
- [Containment](#containment)
- [Overlaps](#overlaps)
- [Regions](#regions)
- [Cameras: rays, picking, projection, view volumes](#cameras-rays-picking-projection-view-volumes)
- [Lines of sight and visibility](#lines-of-sight-and-visibility)
- [Relations](#relations)
- [A compact description of the scene](#a-compact-description-of-the-scene)
- [Caching, history and determinism](#caching-history-and-determinism)
- [JSON requests, the command line and scripts](#json-requests-the-command-line-and-scripts)
- [Errors](#errors)
- [Design](#design)
- [Performance](#performance)
- [Limitations](#limitations)

## Conventions

- **World space.** Positions, directions, normals and distances are in the
  scene's world frame: right-handed, +Z up, in scene units (meters by
  convention). Objects' front is their local -Y (cameras look along it).
  Parents, rotations, non-uniform and mirrored (negative) scales are applied
  exactly as the renderers apply them (the same world matrices).
- **Which objects take part.** Objects with geometry: meshes and procedural
  objects (their current evaluation, modifiers included). Objects that share a
  mesh (copies made by `duplicate`, objects of one procedural node) are
  instances of it and are each queried at their own place. Cameras, lights and
  empties have no surface; region queries treat them as points (their world
  position).
- **Hidden objects** (`obj.visible == False`) are left out unless
  `include_hidden=True`. An object named as the subject of a query (the object
  of `bounds`, `contains_point`, `distance`, `relation`, the target of
  `visible_from`) always takes part.
- **Selecting objects.** `objects=` restricts a query to some objects and
  `exclude=` leaves some out: an object, an id, a name, or a list of them.
- **Results** are frozen dataclasses with a `to_dict()` (their JSON form). They
  name objects by id and name (`hit.object.id`, `hit.object.name`); prefer ids
  for references that must survive renames.
- **Order.** Lists are in a defined order: ray hits by distance, nearest objects
  by distance, and everything else in scene order (depth first through the
  hierarchy, as `for obj in scene` yields them). Ties are broken by scene order,
  then by triangle index.
- **Exactness.** Answers come from the triangles: float64 arithmetic on the
  meshes' float32 positions carried into world space. Bounding boxes only select
  candidates; a result that rests on boxes alone says so (`exact=False`, the
  region relation `"bounds"`). Closed triangles are closed sets: touching counts
  as touching.

## Tags and metadata

Objects carry **tags** (a sorted set of short strings) and **metadata** (a JSON
object) for discovery and for tools:

```python
keep = scene.add_cube(name="Keep", size=4)
keep.tags = ["castle", "stone"]
keep.add_tags("landmark")
keep.metadata = {"floors": 3, "doors": [{"side": "south"}]}
keep.set_metadata("owner", "the baron")
keep.remove_metadata("doors")
print(keep.has_tag("castle"), keep.metadata)
```

- Tags: 1 to 64 characters, no surrounding whitespace, at most 64 per object.
- Metadata: string keys; values are JSON (strings, finite numbers, booleans,
  `None`, lists, objects); at most 16 KiB as JSON. `obj.metadata` returns a
  copy: change entries with `set_metadata`/`remove_metadata` or assign a whole
  new dict.
- Every change is an undo step. Tags and metadata are saved in projects (format
  6, [FILE_FORMAT.md](FILE_FORMAT.md)) and history files, written to glTF
  (`node.extras.aether3d`, [GLTF.md](GLTF.md)) and read back, and copied by
  `duplicate` and `merge`.
- Scene scripts set them with `set_object` (`tags`, `add_tags`, `remove_tags`,
  `metadata`; a `null` value removes an entry); the command line with
  `aether3d object set PROJECT OBJECT --tags a,b --add-tag c --remove-tag d
  --metadata floors=3`.

## Bounds

`scene.bounds(obj=None, space="world", children=False, include_hidden=False)`
and `obj.bounds(...)` return an `ObjectBounds` (or `None` without geometry):

| Field | Meaning |
| --- | --- |
| `aabb` | the axis-aligned box (`min`, `max`, `center`, `size`) of the geometry's vertices: exact |
| `obb` | world space: the object's local box carried through its world matrix (`center`, `half_axes`, `half_extents`, `orthogonal`, `corners()`); under a parent's non-uniform scale and rotation it is a parallelepiped (`orthogonal` false) |
| `sphere` | a sphere enclosing every vertex, centered on the box (it encloses; it is not the smallest one) |
| `objects`, `vertices`, `triangles` | what contributed |

`space="local"` gives the box in the object's own coordinates (before its
transform). With `children=True` the object's visible descendants contribute
too (hidden ones with `include_hidden`): one box around a hierarchy, computed
in the object's local frame for `obb` and `space="local"`. `scene.bounds()` is
the box of the whole scene.

## Finding objects

`scene.find_objects(**filters)` returns the objects that pass every filter, in
scene order:

| Filter | Matches |
| --- | --- |
| `name` | the exact name, or a case-sensitive glob (`"Tower*"`, `"Leaves ?"`) |
| `kind` | `"mesh"`, `"procedural"`, `"camera"`, `"light"`, `"empty"`, `"geometry"` (mesh or procedural), or a list |
| `tags` / `any_tags` | has every one / at least one of these tags |
| `metadata` | has these entries, equal as JSON (`3 == 3.0`, but `true` is not `1`) |
| `metadata_keys` | has these keys |
| `parent` / `ancestor` | is a direct child / a descendant of this object |
| `visible` | `True` or `False` |
| `material` | uses this material (its own or a face group's) |
| `region`, `region_mode` | is in this [region](#regions), hidden objects included |

## Rays

`scene.raycast(origin, direction, *, max_distance=inf, min_distance=0,
objects=None, exclude=None, include_hidden=False, surfaces="geometry")` returns
the first surface along the ray (a `SurfaceHit`, or `None`), and
`scene.raycast_all(..., limit=None)` every surface, nearest first.

- `direction` need not be unit length; distances are along the normalized ray.
  Hits count when `min_distance < distance < max_distance`: a surface through
  the origin itself is not hit (rays can start on a surface).
- **Shared edges and vertices.** Rays are tested with the watertight test of the
  Stage 8 BVH (Woop et al. 2013): a ray never slips between adjacent triangles.
  A ray through an edge or vertex shared by several triangles of one object hits
  it once (the triangle with the lowest index). Rays parallel to a face and in
  its plane do not hit that face; they hit its edges' neighbors when they cross
  them.
- `raycast` is the first hit of `raycast_all`, by construction.

A `SurfaceHit` (also the result of closest-point queries):

| Field | Meaning |
| --- | --- |
| `object` | the object hit (`id`, `name`) |
| `position`, `distance` | the world point (interpolated on the triangle) and its distance along the ray |
| `normal` | the triangle's unit geometric normal on its front side: the side its corners wind counterclockwise around in the mesh's coordinates (mirrored transforms keep outward normals outward) |
| `shading_normal` | the mesh's vertex normals interpolated and carried to world space (inverse transpose), or `normal` without them |
| `front_face` | whether the ray arrived from the front side (`None` for closest points) |
| `triangle`, `face_group`, `barycentric` | the triangle in the object's (evaluated) mesh, its face group, the weights of its corners 0, 1, 2 |
| `uv` | interpolated texture coordinates (`None` without UVs) |
| `material` | the material of the triangle's face group (`None`: the object's own color) |
| `instance_of`, `instance_count` | the first object (in scene order) sharing this mesh, and how many share it |

**Rendered surfaces.** `surfaces="geometry"` (default) treats every triangle
as solid from both sides. `surfaces="rendered"` lets rays through what the
renderers do not draw: texels below an alpha-masked material's cutoff, the backs
of single-sided materials (and, for `pick`, faces removed by the render
settings' `cull`). It uses the path tracer's own rules (`PathScene.trace`), so
cut-out leaves are transparent to rays exactly where the path tracer lets light
through. Hidden objects never render, so they cannot be queried as rendered
surfaces.

## Nearest points and distances

- `scene.closest_point(point, *, max_distance=inf, objects=..., exclude=...,
  include_hidden=False)`: the nearest surface point and the point-to-surface
  distance (`SurfaceHit`; `None` if nothing is within `max_distance`).
- `scene.nearest_objects(point, *, count=None, max_distance=inf, ...)`: objects
  by surface distance, nearest first, each with its nearest point.
- `scene.distance(a, b)`: the shortest distance between two objects' surfaces
  (`ObjectDistance`: `distance`, `point_a`, `point_b`, `intersecting`). Zero
  when the surfaces touch or cross; both points are then one common point.

Closest points on triangles are exact (Ericson, *Real-Time Collision Detection*,
5.1.5; degenerate triangles fall back to their edges). Distances between
triangles are exact: zero if they intersect, else the least vertex-triangle or
edge-edge distance. Both are pruned with per-object world-space BVHs (branch and
bound: a box's farthest corner bounds the distance to what is inside it).

## Containment

`scene.contains_point(obj, point, *, tolerance=1e-6)` returns a `Containment`:

| `state` | When |
| --- | --- |
| `"surface"` | the point is within `tolerance` of the surface (any mesh) |
| `"inside"` / `"outside"` | the mesh is a **closed solid**, decided by its winding number |
| `"undefined"` | the mesh bounds no solid: it is open (boundary edges), non-manifold (an edge shared by more than two triangles) or inconsistently oriented; `reason` says which |

- A closed solid is watertight, edge-manifold and consistently oriented, with
  vertices at exactly equal positions welded (as `inspect_mesh` reports it);
  face-split meshes such as the built-in cube are closed.
- `winding` is the generalized winding number (the triangles' signed solid
  angles over 4π; Van Oosterom and Strackee 1983), corrected for mirroring
  transforms: about 1 inside an outward-facing solid, 0 outside, and 2 inside two
  nested shells of the same orientation. For open meshes it is a fraction that
  measures how much of the surface surrounds the point (negative in front of a
  surface's front side), but no inside or outside is claimed.
- `distance` is the point's distance to the surface.

## Overlaps

`scene.overlaps(objects=None, *, against=None, exclude=None,
include_hidden=False, exact=True, tolerance=0.0)` returns pairs of objects, in
scene order (`Overlap`: `a`, `b`, `bounds_overlap`, `intersecting`, `contained`,
`exact`):

1. **Broad phase:** the pairs whose world boxes (grown by `tolerance`) overlap,
   from a BVH over the objects' boxes. These are candidates, not collisions:
   with `exact=False` they are returned as they are (`intersecting` is `None`,
   `exact` false).
2. **Narrow phase (default):** each candidate is decided on its triangles (a BVH
   pair walk). The pair is reported when the surfaces share a point
   (`intersecting`; with `tolerance > 0`, when they come within it), or when one
   lies entirely inside the other's closed solid without touching it
   (`contained`: `"a_in_b"` or `"b_in_a"`). Disjoint candidates are dropped.

Triangle pairs are tested exactly: a separating axis (both normals and the nine
edge cross products) proves a pair apart; every other pair is decided
constructively (an edge of one crossing the other, or a zero vertex-triangle or
edge-edge distance, within 1e-12 of the coordinates' scale: the rounding of
exact contacts). Containment needs one point per connected component of the
inner object, tested against the outer solid.

## Regions

`scene.query_region(region, *, mode="intersects", objects=..., exclude=...,
include_hidden=False)` takes a `Box(min, max)` (an `AABB`) or a
`Sphere(center, radius)` from `aether3d.spatial`, both closed, and returns
`RegionMatch`es (`object`, `relation`, `exact`) in scene order:

| `mode` | Matches, with their `relation` |
| --- | --- |
| `"intersects"` (default) | `"inside"`: every vertex in the region (the region is convex, so the whole object is); `"intersects"`: the surface meets the region; `"contains"`: the region lies inside the object's closed solid without touching its surface |
| `"inside"` | only `"inside"` |
| `"bounds"` | `"bounds"`: the world box meets the region (candidates, `exact` false) |

Triangles are tested against boxes with the separating axis test (the box's
three axes, the triangle's normal, nine edge cross products) and against spheres
by their closest point. Objects without geometry are `"inside"` when their
position is.

## Cameras: rays, picking, projection, view volumes

Image points are in pixels from the image's top-left corner: pixel `(i, j)`
covers `[i, i + 1) x [j, j + 1)` and its center is `(i + 0.5, j + 0.5)`.
`camera` is a camera object, id or name (default: the active camera); `width` x
`height` is the image size (default: the scene's render settings), which sets
the aspect ratio as in renders.

- `scene.camera_ray(x, y, ...)`: the ray through an image point (`CameraRay`:
  `origin`, unit `direction`, and the near and far planes as distances along
  it). It is, bit for bit, the ray the path tracer traces through that point.
- `scene.pick(x, y, *, camera=None, width=None, height=None, objects=None,
  exclude=None, surfaces="rendered", cull=None)`: the surface the camera shows
  there (`SurfaceHit` or `None`), between its near and far planes. By default
  it skips what is not rendered, including faces culled by the render settings.
  Away from silhouettes it picks the object the rasterizer draws in that pixel
  (`result.object_index`); the tests and `examples/hero_picking.py` check it.
- `scene.project(points, ...)`: where world points appear (`ScreenPoint`: `x`,
  `y`, `depth` along the view direction, `in_view`); one point or a list. A
  perspective camera cannot project points at or behind its eye (`x` and `y` are
  `None`).
- `scene.objects_in_view(..., exact=True)`: objects in the view volume (the
  frustum between the near and far planes), as `ViewMatch`es with
  `visibility` `"inside"` or `"partial"`. Exact: triangles against the frustum
  by separating axes. `exact=False` returns the objects whose world box meets
  it. Occlusion is not considered: see `visible_from` and `pick`.

## Lines of sight and visibility

- `scene.line_of_sight(start, end, *, objects=..., exclude=..., include_hidden=False,
  surfaces="geometry", margin=1e-6)`: whether nothing lies on the segment
  (`LineOfSight`: `clear`, `distance`, `blocker`: the first surface on the way).
  Ends may be points or objects (their world position; such objects never
  block). Surfaces within `margin` times the segment's length of either end do
  not block, so segments may start and end on surfaces.
- `scene.visible_from(target, observer, *, samples=64, ...)`: how much of the
  target's surface an observer sees (`Visibility`: `samples`, `visible`,
  `fraction`, `blockers` with their counts, `outside_view`). `samples` points
  are spread over the target's surface deterministically (proportionally to
  area, an R2 sequence inside triangles: `aether3d.spatial.surface_samples`) and
  joined to the observer by lines of sight; the target itself blocks its far
  side. The observer is a point or an object; a camera also counts samples
  outside its view volume as not visible.

## Relations

`scene.relation(a, b, *, near_distance=None, tolerance=1e-6, viewer=None)`
returns a `Relation` of object `a` to object `b`. Every predicate is exact for
its definition (tolerance in world units):

| Field | Definition |
| --- | --- |
| `distance` | the shortest distance between the surfaces ([`distance`](#nearest-points-and-distances)) |
| `intersecting` | `distance <= tolerance` |
| `near` | `distance <= near_distance` (`None` without `near_distance`) |
| `above` | `a`'s lowest vertex is at or above `b`'s highest (`a.min_z >= b.max_z - tolerance`) and their footprints (world boxes on the XY plane) overlap; `below` the reverse |
| `inside` | `a` lies inside `b`'s closed solid without touching its surface (`None` if `b` is not a closed solid); `contains` the reverse |
| `in_front_of` | `a` lies entirely beyond `b`'s front: its least extent along `b`'s world front direction (local -Y) is at or past `b`'s greatest; `behind` the reverse |
| `viewer` | with `viewer`: `"a_nearer"`, `"b_nearer"` or `"overlapping"`, comparing depth ranges along a camera's view direction, or distance ranges from a point |

## A compact description of the scene

`scene.describe(include_hidden=True)` (`aether3d query describe PROJECT`) is a
compact JSON summary for tools and agents:

```json
{
  "format": "aether3d.spatial-scene", "version": 1, "scene": "Yard", "revision": 5,
  "frame": {"handedness": "right", "up": [0, 0, 1], "front": [0, -1, 0], "units": "..."},
  "active_camera": "<id>",
  "bounds": {"min": [-10, -10, 0], "max": [10, 10, 1]},
  "objects": [
    {"id": "<id>", "name": "Crate", "kind": "mesh", "parent": null,
     "tags": ["movable", "prop"], "position": [2, 0, 0.5],
     "bounds": {"min": [1.5, -0.5, 0], "max": [2.5, 0.5, 1]},
     "triangles": 12, "closed": true}
  ]
}
```

`visible` appears only when false, `tags`, `metadata` and `material` only when
present; `bounds`, `triangles` and `closed` only for objects with geometry.

## Caching, history and determinism

- **Queries change nothing.** No history step, no change of `scene.revision`, the
  same project document before and after (tested for every query).
- **One index per scene revision.** The world-space data (vertices, per-object
  BVHs, the two-level BVH, topology) is built on first use and reused while
  `scene.revision` is unchanged. Every change increases the revision: edits,
  undo and redo included, so a cached index never outlives what it describes.
  Inside an open operation or transaction (`scene.history.in_progress`), queries
  see the changes made so far and build a fresh index; after a rollback the
  scene is back at its revision and the cached index is valid again. The cache
  is weak: it goes with the scene.
- **Deterministic.** The same scene and arguments give the same results, bit for
  bit, run after run and in a fresh copy of the scene: traversal orders and tie
  breaks are fixed, and sampling is a fixed sequence.

## JSON requests, the command line and scripts

Every query is also a JSON request with the Python argument names:

```python
from aether3d import run_query

run_query(scene, {"query": "raycast", "origin": [2, 0, 10], "direction": [0, 0, -1]})
# {"query": "raycast", "result": {"object": {"id": ..., "name": "Crate"}, ...}}
```

| Request `query` | Python | Request `query` | Python |
| --- | --- | --- | --- |
| `describe` | `scene.describe` | `region` | `scene.query_region` |
| `bounds` (`object`) | `scene.bounds` | `camera_ray` | `scene.camera_ray` |
| `find` | `scene.find_objects` | `pick` | `scene.pick` |
| `raycast`, `raycast_all` | `scene.raycast`, `scene.raycast_all` | `project` (`points`) | `scene.project` |
| `closest_point` | `scene.closest_point` | `in_view` | `scene.objects_in_view` |
| `nearest` | `scene.nearest_objects` | `line_of_sight` | `scene.line_of_sight` |
| `contains` (`object`, `point`) | `scene.contains_point` | `visible_from` | `scene.visible_from` |
| `distance`, `overlaps` | `scene.distance`, `scene.overlaps` | `relation` | `scene.relation` |

Objects are ids or names, points are `[x, y, z]`, regions are `{"box": {"min":
[...], "max": [...]}}` or `{"sphere": {"center": [...], "radius": r}}`, and a
`max_distance` of `null` means unlimited. Unknown or missing arguments are
`INVALID_REQUEST` errors that list the accepted ones.

**Command line.** `aether3d query QUERY PROJECT KEY=VALUE ...` answers one query
without changing the project (`.glb`/`.gltf` files work too). Values are JSON,
comma-separated numbers, or plain text:

```bash
aether3d query raycast castle.aether origin=0,0,30 direction=0,0,-1 --json
aether3d query nearest castle.aether point=2,3,1 count=3
aether3d query find castle.aether tags=tree --json
aether3d query region castle.aether 'region={"box":{"min":[-5,-5,0],"max":[5,5,9]}}'
aether3d query pick castle.aether x=960 y=540 width=1920 height=1080 --json
aether3d query run castle.aether --request requests.json --json   # one request or a list
```

With `--json` the result is wrapped in the CLI's usual envelope (`{"ok": true,
"command": "query", "query": ..., "result": ...}`); errors come back as
`{"ok": false, "error": {"type": "QueryError", "code": ..., "location": ...}}`
with exit code 1.

**Scene scripts** run a query with the read-only `query` operation; it sees the
scene as the operations before it left it, and adds nothing to the script's
history step:

```json
{"op": "query", "query": "raycast", "origin": [0, 0, 10], "direction": [0, 0, -1]}
```

The tests check that Python, `run_query`, the command line and scripts give the
same JSON for every query.

## Errors

Malformed queries and queries that cannot be answered raise `QueryError` (an
`AetherError` and a `ValueError`) with a stable `code`, the offending argument
as `location`, and often a `hint`:

| Code | When |
| --- | --- |
| `INVALID_ARGUMENT` | a value has the wrong type or range (`location` names it) |
| `INVALID_REQUEST` | a JSON request names an unknown query, misses or adds arguments |
| `OBJECT_NOT_FOUND` | no object with that id or name (or one from another scene) |
| `NO_GEOMETRY` | the object has no surface (a camera, light or empty) |
| `NO_CAMERA`, `NOT_A_CAMERA`, `INVALID_CAMERA_TRANSFORM` | camera queries without a usable camera |
| `MATERIAL_NOT_FOUND` | `find_objects(material=...)` names no material |
| `SINGULAR_TRANSFORM` | local bounds of a hierarchy under a zero-scale object |
| `GEOMETRY_EVALUATION_FAILED` | a procedural object's geometry fails to evaluate (`location`: `objects[NAME]`) |

## Design

- `aether3d.spatial.index`: the `SpatialIndex` of a scene: every object with
  geometry, its world matrix (the renderers' `world_matrices`), and, built on
  demand, world-space vertices, a world-space BVH over its triangles, node
  boxes, connected components and topology. For rays it builds the Stage 8
  two-level `SceneAccel` over the objects asked about; its bottom levels come
  from the renderers' shared, content-keyed BLAS cache.
- `aether3d.spatial.rays`: the closest-hit traversal is `SceneAccel.closest_hit`;
  `all_hits` walks the same structure without shrinking the ray and tests
  triangles with the same watertight test on the same object-space rays (the
  same arithmetic), so the nearest of all hits is the closest hit.
- `aether3d.spatial.geometry`: the exact kernels, vectorized with NumPy: closest
  points on triangles, segment distances, triangle-triangle intersection and
  distance, triangle-versus-convex-polyhedron separating axes, winding numbers,
  connected components, BVH box queries.
- `aether3d.spatial.queries`: the queries; `aether3d.spatial.types`: regions and
  results; `aether3d.spatial.request`: JSON requests (CLI and scripts).

The two-level BVH's boxes are closed: a ray with a zero direction component that
lies exactly in the plane of a box's face is inside the box's slab (boxes are
grown by one ulp at build time; see [RENDER_QUALITY.md](RENDER_QUALITY.md)).

## Performance

Measured on the development machine (i9-12900H, one process, NumPy), with a
warm index unless noted:

| Scene | Query | Time |
| --- | --- | --- |
| The procedural castle (21 objects with geometry, 6,804 triangles) | first `raycast` (evaluates and indexes) | 21 ms |
| | `raycast` / `raycast_all` | 1.3 / 1.8 ms |
| | `closest_point` / `nearest_objects(count=5)` | 1.6 / 6 ms |
| | `overlaps()` (all pairs, exact) | 38 ms |
| | `objects_in_view` / `query_region` | 7.5 / 2.2 ms |
| | `visible_from` (64 samples) / `relation` | 5 / 8.6 ms |
| 300 instanced spheres (662,402 triangles) | first `raycast` / then | 16 / 2.8 ms |
| | `closest_point` / `nearest_objects(count=10)` | 64 / 56 ms |
| | `overlaps()` / `query_region` | 95 / 23 ms |
| | `objects_in_view` | 206 ms |
| | 100 `pick`s / `visible_from` (256 samples) | 136 / 10 ms |

Queries run in Python with vectorized NumPy kernels; they are meant for tools,
editors and agents, not for per-frame game logic on large scenes.

## Limitations

- **Sampling where stated.** `visible_from` estimates the visible fraction from
  a fixed set of surface samples (exact for each sample); the bounding sphere of
  `bounds` encloses but is not the smallest.
- **Box-based predicates where stated.** `above`/`below` compare world-box
  heights and footprints: an object on a slope or under an overhang is judged by
  its box, not its contact.
- **Alpha masks** (`surfaces="rendered"`) are evaluated at the exact hit point
  at full texture resolution, as the path tracer does without a ray footprint;
  at the edge of a cut-out a rasterized pixel (filtered over its footprint) may
  disagree.
- **Containment** is only claimed for closed solids; vertices are welded only
  at exactly equal positions, so a mesh with near-miss seams is open and its
  containment is `"undefined"`.
- **Zero-scale objects** (singular world matrices) are never hit by rays (as in
  renders) but still have bounds and closest points. Objects with non-finite
  world matrices take no part.
- **No physics.** Overlaps and distances are geometric tests on static geometry:
  there is no collision response, motion or time.
- **Textures for rendered surfaces** are read when the index is first built for
  a revision; an image file changed on disk without a scene change is seen
  after the next change.
- **Not thread-safe:** query a scene from one thread at a time.
