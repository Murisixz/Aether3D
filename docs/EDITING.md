# Mesh editing

Aether3D 0.3.0 adds a programmable polygon-modeling engine: create, select,
transform and reshape vertices, edges and faces from Python or from JSON
scripts on the command line, then commit the result back to a scene object
as an ordinary immutable `Mesh`.

```python
from aether3d import Scene

scene = Scene()
crate = scene.add_cube("Crate", size=2)

with crate.edit_mesh() as edit:  # an EditSession
    top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
    edit.inset_faces(top, 0.2)
    edit.extrude_faces(top, -0.1)  # a recessed lid
    edit.bevel_edges(edit.select_edges(boundary=False, nearest=(1, 1, 0)), 0.05)
    print(edit.validate().status)  # "ok"
# leaving the block commits a new Mesh to crate (only to crate)
```

## Contents

1. [Topology representation](#topology-representation)
2. [Conventions](#conventions)
3. [Element ids](#element-ids)
4. [Sessions, transactions and history](#sessions-transactions-and-history)
5. [Selection](#selection)
6. [Operations](#operations)
7. [Attributes: UVs and normals](#attributes-uvs-and-normals)
8. [Face groups](#face-groups)
9. [Validation](#validation)
10. [Edit scripts (JSON) and the CLI](#edit-scripts-json-and-the-cli)
11. [Performance](#performance)
12. [Limitations](#limitations)

## Topology representation

**Choice: BMesh-style topology in id-keyed tables.**

Three options were evaluated against the requirements:

- arbitrary polygons;
- non-manifold input, since glTF imports may contain it;
- per-corner attributes (UV and normal seams);
- stable element references;
- atomic rollback;
- reasonable speed in Python.

| Representation | Strengths | Why it was not chosen |
| --- | --- | --- |
| Half-edge (pointer-linked) | O(1) adjacency traversal; elegant for manifold surfaces | Cannot represent non-manifold edges or vertices, which real imported assets contain and which must be reported, not rejected or corrupted. Millions of small Python objects are slow and memory-hungry. |
| Full BMesh (pointer cycles: disk and radial lists) | n-gons, non-manifold, per-corner "loops" (Blender's proven model) | The semantics are exactly what is needed, but pointer cycles in Python are slow, and difficult to journal for exact rollback. |
| Indexed polygon soup (face lists only) | Simple and compact | No edge identity, so edges cannot be selected stably; adjacency must be rebuilt for every query. |

`EditableMesh` (`src/aether3d/editing/core.py`) keeps BMesh's semantics but
stores them in integer-keyed tables:

| Element | Stored as |
| --- | --- |
| **Vertex** | Position in a growable `float64` NumPy array indexed by vertex id, plus an alive mask and the set of incident edge ids (the *disk*). |
| **Edge** | An unordered vertex pair `(a, b)` with `a < b`, an id, and the ordered list of faces using it (the *radial* list): 1 face = boundary, 2 = manifold, more = non-manifold. |
| **Face** | An ordered vertex loop with per-corner attribute values (UV, normal, ...) stored alongside, plus per-face data (smooth/flat). |
| **Edge data** | A per-edge *sharp* flag. |

Edges exist exactly while at least one face uses them. Arbitrary n-gons,
open boundaries, non-manifold edges and vertices, and attribute seams are all
representable. Positions in one array make transforms vectorized. All
mutation goes through seven journaled primitives (add/delete vertex, move
vertices, add/delete/reshape face, set corner values), which gives exact
rollback and undo without copying the mesh.

The immutable indexed-triangle `Mesh` remains the representation for
persistence (`.aether`), inspection and glTF. `EditableMesh.from_mesh()` and
`to_mesh()` convert between the two; `Mesh` itself is unchanged.

## Conventions

| Topic | Convention |
| --- | --- |
| Coordinates | Object-local, right-handed, +Z up (as everywhere in Aether3D). `space="world"` arguments use the object's world transform. |
| Winding | A face's vertex loop runs counter-clockwise seen from its front; the front normal follows the right-hand rule (Newell's method, robust for n-gons). |
| Face normal | `face_normal(f)`: the unit Newell normal (zero for a zero-area face). |
| Corner normals | Per-corner unit vectors used for shading (see [Attributes](#attributes-uvs-and-normals)). |
| UVs | Per-corner, bottom-left origin (`v` up). |
| Edge orientation | `edge_vertices(e)` is `(a, b)` with `a < b`. Where an operation needs a direction (`split_edges`, `loop_cut factor=`), it runs from the lower to the higher vertex id unless stated otherwise. |
| Ordering | Selections are sets and results are sorted; no result depends on hash or iteration order. |

## Element ids

- Vertices, edges and faces have non-negative integer ids, unique per kind
  within one `EditableMesh`.
- **Untouched elements never change id.** An operation that reshapes a face
  keeps the face's id; it is listed in the result's `modified_faces`.
- **New elements get fresh ids** that no earlier element of that kind had.
- **Every operation reports what it did:** `result.created` and
  `result.removed` list ids by kind, derived from the operation's journal, so
  they are complete. Operation-specific mappings are in `result.data`; for
  example extrusion's `{"vertices": {old_id: new_id}}` and `"side_faces"`, or
  subdivision's `{"faces": {original: [pieces]}}`.
- **Undo restores ids exactly.** Ids of elements created by an undone
  operation may be assigned again by later operations, and `redo()`
  reproduces the same ids.
- **An edge is identified by its vertex pair while it exists.** If all its
  faces are removed and the pair is later joined again, that is a new edge
  with a new id.
- **Ids are not persistent across commits.** `to_mesh()` produces an indexed
  triangle mesh. `from_mesh()` assigns ids deterministically from it:
  vertices by first occurrence in the vertex order (after welding), and faces
  by triangle order. So opening the same `Mesh` twice yields the same ids,
  which JSON scripts can rely on across CLI runs.
- **Using an id that does not exist raises `INVALID_ELEMENT`,** listing the
  ids. Nothing is ever silently skipped.

## Sessions, transactions and history

- `obj.edit_mesh(weld="exact", quads=True)` returns an `EditSession`.
  - The session converts the object's mesh into an `EditableMesh`. Objects
    without a mesh start empty.
  - `weld="exact"` merges vertices with bit-identical positions, so that seams
    which only exist to carry different UVs or normals become shared topology
    with per-corner attributes.
  - `quads=True` rebuilds quads from pairs of coplanar triangles whose UVs and
    normals agree along the shared edge (exactly `join_triangles`). This is
    lossless: primitives become editable quads and the surface is unchanged.
- Operations and queries are available on the session itself
  (`edit.extrude_faces(...)`) and on `edit.mesh`.
- **Atomic operations.** Every operation runs in a transaction. If it raises,
  for any reason including bugs, the journal is replayed backwards and the
  mesh is bit-identical to before. The session stays usable.
- **History.** `undo()` and `redo()` replay or re-run operations exactly; up to
  `EditableMesh.max_history` (256) operations are kept. `history` lists the
  applied operations with their results. This is a per-session undo stack,
  not the persistent modifier history planned for a later stage.
- **Commit.** `commit(scope="object", allow_invalid=False)`:
  1. Validates the mesh and refuses error-severity findings (`VALIDATION_FAILED`;
     the session stays open).
  2. Refuses if the object's mesh was replaced meanwhile (`STALE_SESSION`).
  3. Triangulates the polygons.
  4. Assigns the new immutable `Mesh` and closes the session.

  The `CommitResult` contains the mesh, the validation report, a Stage 2
  `inspect_mesh` report of the result, and diagnostics.
- **No-op commits** (no operation applied) keep the original `Mesh` object, so
  instancing is untouched.
- **Instancing is preserved.** By default only the edited object receives the
  new mesh; other objects sharing the old one keep it. `scope="shared"`
  updates every object in the scene that used the same mesh.
- **Context manager.** `with obj.edit_mesh() as edit:` commits when the block
  ends normally and discards everything if it raises.
- **Closed sessions.** After a commit or discard, operations raise
  `SESSION_CLOSED`.
- **Standalone use.** `EditableMesh` also works on its own:
  `EditableMesh.from_mesh(mesh)` ... `to_mesh()`.

## Selection

All queries return an immutable `Selection` (sets of vertex, edge and face
ids) and combine with `|`, `&` and `-`. Criteria in one call are ANDed.

```python
edit.select_faces(normal=((0, 0, 1), 5))  # within 5 degrees of +Z
edit.select_vertices(box=((-1, -1, 0), (1, 1, 2)))
edit.select_edges(sphere=((0, 0, 1), 0.1), boundary=False)
edit.select_faces(plane={"point": (0, 0, 0), "normal": (1, 0, 0), "side": "front"})
edit.select_faces(sides=3)  # all triangles
edit.select_vertices(nearest=(1, 1, 1))  # exactly one, or AMBIGUOUS_SELECTION
edit.select_faces(normal=((1, 0, 0), 1), expect=1)  # SELECTION_COUNT_MISMATCH otherwise
edit.select_faces(box=..., test="all")  # all vertices inside (also "any")
edit.select_vertices(predicate=lambda vid, xyz: xyz[2] > 0)  # Python only
edit.select_faces(normal=((0, 1, 0), 1), space="world")  # through the object's transform
```

| Query | Result |
| --- | --- |
| `select_all()` | Everything |
| `edge_loop(e)` | The loop through `e`: straight through vertices with four manifold edges; stops at poles and boundaries. A boundary edge yields its boundary loop. |
| `edge_ring(e)` | Edges opposite each other across consecutive quads |
| `linked_faces(faces)` | All faces connected through edges |
| `grow(sel)` | Adds neighbouring elements |
| `region_boundary(faces)` | The outline edges of a face region |
| `boundary_edges()`, `non_manifold_edges()` | As named |
| `to_vertices`, `to_edges`, `to_faces` | Convert between kinds; `to_faces` takes faces whose vertices are all selected |

**Safety rules:**

- `nearest=` never picks arbitrarily: ties within 1e-9 of the scale raise
  `AMBIGUOUS_SELECTION`, listing the tied ids.
- `expect=n` raises `SELECTION_COUNT_MISMATCH` when the count differs.
- Operations raise `EMPTY_SELECTION` for empty input.
- Operations raise `WRONG_ELEMENT_KIND` when a selection holds only another
  kind (for example vertices given to `extrude_faces`).

## Operations

Every operation is an atomic, undoable method returning an `OperationResult`
(`created`, `removed`, `modified_faces`, `data`, `diagnostics`). The full list
is `EditableMesh.operations()`. Errors are `EditError` with a stable `code`,
the offending `elements`, and a `hint`.

### Creation and deletion

| Operation | What it does | Notable errors |
| --- | --- | --- |
| `add_box(size, center=)` | A closed box of 6 quads, 8 shared vertices, per-face UVs (same layout as `create_cube`) | |
| `add_grid(x_segments, y_segments, size, center=)` | A flat quad grid in XY facing +Z, UVs spanning 0..1 | |
| `create_vertex(position, space=)` | A loose vertex | |
| `create_face(vertices, smooth=, allow_non_manifold=False)` | A polygon from existing vertices in counter-clockwise order. Corner UVs are inherited where a vertex has one consistent UV; otherwise `UVS_UNDEFINED` is reported. | `INVALID_POLYGON`, `DEGENERATE_RESULT`, `SELF_INTERSECTING_FACE`, `DUPLICATE_FACE`, `NON_MANIFOLD_RESULT` (third face on an edge), `INCONSISTENT_WINDING` |
| `delete_faces`, `delete_edges`, `delete_vertices` | Delete elements (and the faces using them); vertices left loose are removed unless `keep_loose_vertices=True` | |

### Transforms

`translate(elements, offset)`, `rotate(elements, angle, axis, pivot=)`,
`scale(elements, factor, pivot=)`, `transform(elements, matrix)`,
`set_positions({id: xyz})` and `move_along_normals(elements, distance)`.

- `elements` is a `Selection` (its vertices, edges and faces all move) or a
  list of vertex ids.
- `pivot` is `"center"` (the default, the mean of the moved vertices),
  `"origin"`, or a point.
- `space="world"` interprets offsets, axes, pivots and matrices in world space.
  The edit is converted exactly as `M⁻¹·A·M`, so it is correct under rotated,
  non-uniformly scaled and mirrored parents.
- **Mirroring transforms** (negative determinant) re-wind the faces of every
  connected piece that moves entirely, so mirrored pieces stay outward-facing.
  Mirrored parts still attached to unmoved faces keep their winding, which
  keeps orientation consistent. This is reported as `WINDING_REVERSED`.
- **Zero-area faces** produced by a transform raise `DEGENERATE_RESULT` unless
  `allow_degenerate=True`.
- **Singular world transforms** make world-space edits raise
  `SINGULAR_WORLD_TRANSFORM`.

### Modeling operations

| Operation | Behaviour and ids | Fails cleanly with |
| --- | --- | --- |
| `extrude_faces(faces, distance \| offset=, mode="region"\|"individual")` | **Region:** each connected group moves along its area-weighted normal (or by `offset`), with walls along its outline. **Individual:** every face is extruded on its own. Extruded faces keep their ids. `data["side_faces"]` are the walls; `data["vertices"]` maps old boundary vertices to their new copies. Works on open surfaces too. A closed shell has no outline, so it is moved (`EXTRUDE_WITHOUT_BOUNDARY`). | `ZERO_EXTRUSION`, `DEGENERATE_RESULT` (e.g. an offset parallel to a wall), `INCONSISTENT_WINDING`, `NOT_MANIFOLD` |
| `inset_faces(faces, thickness, depth=, mode="individual"\|"region")` | Shrinks faces inward by `thickness`, measured perpendicular to each edge within the face (mitred corners), and fills the border with quads. `depth` then moves the inset faces along their normal. Region mode insets only the outline of each connected group. | `INSET_TOO_LARGE` (a face would invert, fold or self-intersect, or an edge would reverse), `UNSUPPORTED_TOPOLOGY` (closed shell; an outline touching itself) |
| `bevel_edges(edges, width)` | Chamfer: in every adjacent face the new edge is parallel to the old one at perpendicular distance `width`. Mitres where two bevelled edges share a face; corners slide along unbevelled edges. Holes at corners where three or more points meet are filled (e.g. the triangle at a fully bevelled box corner); degenerate corners merge into the chamfer. `data["faces"]` are the chamfers, `data["corner_faces"]` the fills. | `NOT_MANIFOLD`, `BEVEL_UNSUPPORTED_VERTEX` (open boundary, non-manifold fan, fewer than 3 edges), `BEVEL_WIDTH_TOO_LARGE`, `UNSUPPORTED_OPERATION` (`segments > 1`) |
| `subdivide_faces(faces, cuts=1)` | Shape-preserving. Quads become a `(cuts+1)²` grid (bilinear); triangles become `(cuts+1)²` triangles (barycentric); larger polygons become a fan of quads around their center (`cuts=1` only). Shared edges are split once and neighbours gain the new vertices. `data["faces"]` maps each face to its pieces. | `UNSUPPORTED_TOPOLOGY` (n-gon with `cuts > 1`) |
| `loop_cut(edge, cuts=1, factor=)` | Cuts new edge loops across the ring of quads through `edge` (see `edge_ring`). The ring stops at boundaries and non-quads, whose faces simply gain the vertices; closed rings wrap around. With `cuts=1`, `factor` is measured from the lower-id vertex of `edge`. `data["edges"]` are the new loops. | `NO_QUAD_RING`, `NON_ORIENTABLE_RING` (a Möbius ring), `UNSUPPORTED_TOPOLOGY` (the ring crosses a quad twice) |
| `merge_vertices(vertices, at="center"\|"keep"\|xyz, keep=)` | Merges into one vertex; the survivor is `keep`, or the lowest id. Faces that collapse below 3 vertices are deleted; merged corner attributes are averaged. Zero-area, duplicate or non-manifold results are reported as diagnostics. | `MERGE_INVALID_FACE` (a face would use the vertex twice, e.g. opposite quad corners), `TOO_FEW_ELEMENTS` |
| `merge_by_distance(distance, vertices=None)` | Merges clusters of vertices closer than `distance` (transitively, using a spatial hash) at their centers | as `merge_vertices` |
| `collapse_edges(edges)` | Collapses edges to points; connected edges merge into one vertex at their center | as `merge_vertices` |
| `dissolve_edges(edges)` | Removes edges between two faces, merging the faces into one polygon, which keeps the lower id | `EDGE_NOT_INTERIOR`, `INCONSISTENT_WINDING`, `DISSOLVE_INVALID_FACE` |
| `join_triangles(faces=None, max_angle=1.0)` | Merges adjacent coplanar triangle pairs into convex quads where their UVs and normals agree (so nothing is lost). Best-shaped pairs go first; ties resolve by edge id. | |
| `split_edges(edges, cuts=1)`, `split_edge(edge, factor, from_vertex=)` | Inserts vertices on edges; both adjacent faces gain them | |
| `split_face(face, vertex_a, vertex_b)` | Connects two non-adjacent vertices of a face with a new edge. The part from `vertex_a` to `vertex_b` keeps the id. | `VERTICES_ADJACENT`, `CHORD_OUTSIDE_FACE` (concave faces), `NON_MANIFOLD_RESULT`, `DEGENERATE_RESULT` |
| `rip_vertices(vertices)` | Splits each vertex so every face around it gets its own copy | |
| `separate_faces(faces)` | Disconnects a region from the rest of the mesh by duplicating shared vertices | |
| `triangulate_faces(faces=None)` | Ear clipping that keeps winding and handles concave and collinear vertices; the first triangle keeps the id | `TRIANGULATION_FAILED` |
| `flip_faces(faces)` | Reverses winding and negates stored normals | |
| `set_smooth(faces, smooth)`, `set_sharp(edges, sharp)` | Shading controls (see below) | |

## Attributes: UVs and normals

Per-corner attributes live in **corner layers**: `"uv"` (2D) and `"normal"`
(3D, renormalized after interpolation). Every operation carries every layer
through the same code path, so future per-corner data such as vertex colours
or extra UV sets is added as a new `CornerLayer` with no change to the
operations. Per-face data (the smooth flag) and per-edge data (the sharp
flag) are kept the same way, and so is each face's group (see below).

| Situation | Policy |
| --- | --- |
| Corners an operation does not touch | Preserved exactly, bit for bit. |
| Shape-preserving refinement: `subdivide_faces`, `loop_cut`, `split_edges`, `split_face`, `triangulate_faces`, `join_triangles`, `dissolve_edges` | UVs are interpolated linearly, bilinearly or barycentrically. Stored normals are kept and new ones interpolated, so smooth shading keeps its analytic normals. |
| New corners inside an existing face: `inset_faces`, `bevel_edges` | Attributes come from an affine fit over the original face, which is exact for any UV layout that is affine on a planar face. The border and chamfer quads take matching values, so the texture layout is preserved. |
| Rigid or affine transforms | Normals of faces that move entirely are transformed exactly, by `A⁻ᵀ` or `sign(det)·A⁻ᵀ`. |
| Geometry changed (moved vertices, extrusion, merges, new faces) | Normals of the affected corners are recomputed at commit from the shading model. All other corners keep their stored normals. |
| Unavoidable loss | Reported: extrusion side walls copy the boundary UVs and so have zero UV area (`UV_DEGENERATE_FACES`); a new face whose vertices have no single existing UV gets (0, 0) (`UVS_UNDEFINED`). |

**Shading model, as in Blender.** Faces are flat or smooth, and edges may be
sharp.

- A **flat** face's corners use the face normal.
- A **smooth** corner uses the angle-weighted average of the smooth faces
  around its vertex that can be reached without crossing a sharp, boundary or
  non-manifold edge.
- `from_mesh` infers the flags from the existing normals: a face is smooth if
  its corner normals differ from its face normal, and an edge is sharp if the
  normals of its two faces differ across it.
- New faces inherit the flag of the face they came from.

## Face groups

Every face belongs to a face group, an integer from 0 to 65535 (0 by
default). An object can give each group its own material
(`scene.assign_material(obj, material, group=k)`; see
[MATERIALS.md](MATERIALS.md#face-groups-several-materials-on-one-mesh)).
Groups are a per-face value, carried like the smooth flag:

```python
with crate.edit_mesh() as m:
    lid = m.select_faces(normal=((0, 0, 1), 1), expect=1)
    m.set_group(lid, 1)                     # an undoable operation
    m.inset_faces(lid, 0.1)                 # the inset border joins group 1 too
    print(m.face_group(next(iter(lid.faces))), m.has_groups)
    m.select_faces(group=1)                 # every face of group 1
scene.assign_material(crate, "Lid", group=1)
```

- **Inheritance.** New faces inherit the group of the face they come from:
  extrusion walls, inset borders, subdivision and loop-cut pieces, and split
  and triangulated parts. Ripping and separating keep faces, and so their
  groups. Faces between two faces take one of their groups:
  - a bevel chamfer takes the group of the beveled edge's lower-id face, and
    a bevel corner patch that of the corner's lowest-id face;
  - a face created with `create_face` takes the group of a neighbouring face
    (0 without neighbours).
- **Merging.**
  - `join_triangles` never merges faces of different groups.
  - `dissolve_edges` across two groups keeps the surviving face's group and
    reports `FACE_GROUP_CHANGED`.
- **Round trip.** `from_mesh` reads `Mesh.groups`, and a commit writes them
  back. A mesh without groups stays without them until `set_group` is used.
- **History.** `undo()` and `redo()` restore groups with everything else.
- **Scripts.** `set_group` is available in edit scripts, and `"group": k` is
  a selection criterion.

## Validation

`edit.validate()` returns a `TopologyReport`. Its findings are the Stage 2
`Diagnostic` objects (stable `code`, `severity`, a `location` naming the
element kind, `count`, sample ids). Its `summary` adds the counts, the
face-size histogram, the Euler characteristic, and `closed`, `edge_manifold`,
`vertex_manifold` and `consistently_oriented`. Commits refuse error-severity
findings.

| Check | Code | Severity |
| --- | --- | --- |
| Internal adjacency consistency (faces, edges and vertex disks agree) | `BROKEN_ADJACENCY` | error |
| Non-finite positions | `NON_FINITE_POSITIONS` | error |
| Polygon outline crosses itself | `SELF_INTERSECTING_FACES` | error |
| Zero-area faces | `ZERO_AREA_FACES` | warning |
| Non-planar polygons (deviation > 1e-5 x size) | `NONPLANAR_FACES` | warning |
| Duplicate faces (same vertex set) | `DUPLICATE_FACES` | warning |
| Coincident vertices (exactly equal positions) | `COINCIDENT_VERTICES` | warning |
| Non-manifold edges (more than two faces) | `NON_MANIFOLD_EDGES` | warning |
| Non-manifold vertices (faces form more than one fan) | `NON_MANIFOLD_VERTICES` | warning |
| Inconsistent winding across an edge | `INCONSISTENT_WINDING` | warning |
| Concave polygons (valid) | `CONCAVE_FACES` | info |
| Boundary edges | `BOUNDARY_EDGES` | info |
| Loose vertices | `LOOSE_VERTICES` | info |

**Not evaluated:** intersections between *different* faces, and overlap of
faces that share no vertices. Both need spatial acceleration structures, and
are listed under `not_evaluated` in every report.

**Stage 2 integration.** `inspect_mesh` (and therefore `aether3d geometry`)
now also evaluates vertex manifoldness on the welded triangle mesh
(`vertex_manifold`, `NON_MANIFOLD_VERTICES`). Commit results include a full
`inspect_mesh` report of the committed mesh.

## Edit scripts (JSON) and the CLI

A script is declarative data; no code is ever evaluated, and only the
operations listed by `EditableMesh.operations()` can be called. The same
runner, `aether3d.run_edit_script(scene, script)`, backs the CLI and is
designed to back a future MCP server unchanged.

```json
{
  "format": "aether3d.edit-script",
  "version": 1,
  "edits": [
    {
      "object": "Crate",
      "scope": "object",
      "operations": [
        {"op": "select", "kind": "faces", "normal": [[0, 0, 1], 1], "expect": 1, "as": "top"},
        {"op": "inset_faces", "faces": {"selection": "top"}, "thickness": 0.2},
        {"op": "extrude_faces", "faces": {"selection": "top"}, "distance": -0.1,
         "save": {"walls": "side_faces"}},
        {"op": "bevel_edges", "edges": {"select": {"kind": "edges", "nearest": [1, 1, 0]}},
         "width": 0.05},
        {"op": "validate", "require_ok": true}
      ]
    }
  ]
}
```

- **`edits`** apply to objects given by name or `{"id": ...}`.
  `"create": true` creates a missing object, which starts with an empty mesh.
  `scope`, `weld`, `quads` and `allow_invalid` mirror the Python API.
- **Step types:**
  - `select` (with the criteria above, storing the result `as` a name);
  - selection queries: `select_all`, `edge_loop`, `edge_ring`, `linked_faces`,
    `region_boundary`, `boundary_edges`, `non_manifold_edges`, `grow`,
    `to_vertices`, `to_edges` and `to_faces`;
  - `combine` (`union`, `intersection` or `difference` of named selections);
  - `validate` (optionally `require_ok`), `summary` and `describe`;
  - `undo` and `redo`;
  - or any editing operation with its keyword arguments.
- **Element arguments** accept id lists, `{"selection": name}`, or an inline
  `{"select": {...}}`. Single-element arguments (`edge`, `face`) require
  exactly one element.
- **`"save": {name: key}`** stores result selections, using `"created"` or a
  `data` key.
- **Errors** are `ScriptError` with a `location` such as
  `edits[0].operations[3].distance` and the underlying `code`, `elements` and
  `hint`.
- **The whole script is atomic.** If any step fails, every object keeps its
  original mesh and created objects are removed.
- **Limits:** at most 10,000 operations per script, 100 cuts per operation,
  20 million elements per mesh, and 16 MiB of script text in the CLI.

```console
$ aether3d topology project.aether --object Crate --json            # summary + validation
$ aether3d topology project.aether --object Crate --elements --limit 50
$ aether3d edit project.aether script.json --json                   # edit in place, atomically
$ aether3d edit project.aether script.json --output new.aether
$ aether3d edit project.aether - --dry-run < script.json            # script from stdin
```

**`edit` behaviour:**

- It writes the project only if every step and every commit succeeds, using
  the atomic write from Stage 1.
- It exits with `1` on any failure; the JSON error carries `code`, `location`,
  `elements` and `hint`.
- `topology` exits with `3` if the mesh has error-severity findings.

## Performance

Measured with `python benchmarks/bench_editing.py` on Arch Linux (Python 3.14,
NumPy 2.5) and reported in the Stage 3 pull request. Every operation is
linear in the size of what it touches. Highlights for a 316x316 grid slab
(about 200k triangles, 100k faces):

- **Conversion** (`from_mesh` with quad rebuilding, `to_mesh`, `validate`): a
  few seconds each, linear, about 90 µs per triangle for `from_mesh`.
- **Local operations** (loop cut, extrusion of a region, undo): milliseconds
  to sub-second, with cost proportional to the selection, not the mesh.
- **Optimizations made after profiling:**
  - an O(V)-per-call extent computation made quad rebuilding quadratic. It is
    now an incrementally grown, cached bounding box, which took the large
    `from_mesh` from 232 s to 7.4 s;
  - per-face NumPy calls on 3- and 4-vertex arrays were replaced by plain
    floats, a 2-3x gain in triangulation, validation and selection.

| Operation | Complexity |
| --- | --- |
| `from_mesh`, `to_mesh`, `validate`, `triangulate_faces()` | O(F·k) for F faces of k vertices; ear clipping is O(k²) per n-gon |
| Selection by geometry | O(candidates) |
| Transforms | O(selected vertices + their faces) |
| Extrude, inset, subdivide, loop cut, bevel, merge | O(selection and its neighbourhood) |
| Undo / rollback | O(size of the operation's journal) |

## Limitations

- **Bevel** is single-segment (chamfer) only. It needs manifold edges whose
  vertices have closed, consistently wound fans of at least 3 edges, so
  bevelling at open boundaries is not supported. Where two faces disagree on a
  slide position, their average is used (`BEVEL_UNEVEN_WIDTH`).
- **N-gons are not persisted.** `.aether` and glTF store triangles. Committing
  triangulates, and the next session rebuilds quads (`quads=True`) where
  triangle pairs are coplanar and attribute-consistent. Non-planar quads and
  n-gons come back as triangles.
- **Normals of changed geometry are recomputed,** not interpolated. Custom
  (hand-authored) normals survive only where geometry is untouched.
- **Extrusion side walls get degenerate UVs** (reported); no automatic UV
  unwrapping.
- **No self-intersection test** between different faces, and no automatic
  repair operations. Problems are reported, not fixed.
- **`merge_by_distance` and `move_along_normals`** use plain distances and
  vertex normals. There is no "even thickness" option.
- **No edge slide, knife, bridge, spin, fill, or rounded bevels** in this
  stage.
