# Procedural geometry

Aether3D 0.4 adds a **non-destructive procedural geometry engine**. A geometry
is described by a *definition*: a source (a parametric primitive, a stored
mesh, or another geometry) followed by an ordered **modifier stack**, with
values that can come from typed **parameters** and **expressions**. Objects use
the *evaluated* result. Changing a parameter or a modifier regenerates
everything that depends on it, while the definitions themselves are never
modified by evaluation.

Everything is available from Python, from declarative JSON scripts, and from
the `aether3d procedural` CLI, and every change is validated, atomic and
undoable.

- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Parameters, expressions and rules](#parameters-expressions-and-rules)
- [Primitives](#primitives)
- [Modifiers](#modifiers)
- [Geometry references](#geometry-references)
- [Evaluation, caching and performance](#evaluation-caching-and-performance)
- [Transactions, undo and errors](#transactions-undo-and-errors)
- [Objects: sharing, baking and editing the source](#objects-sharing-baking-and-editing-the-source)
- [Persistence](#persistence)
- [JSON scripts](#json-scripts)
- [Command line](#command-line)
- [Introspection for AI agents](#introspection-for-ai-agents)
- [Error and diagnostic codes](#error-and-diagnostic-codes)
- [Limitations](#limitations)

## Quick start

```python
from aether3d import Expr, Scene

scene = Scene("Tower Demo")
proc = scene.procedural
proc.add_parameter("height", 6.0, min=2, max=30, unit="m")
proc.add_parameter("windows", 4, min=1, max=12)
proc.add_rule("tall_enough", "height >= windows", "one metre of tower per window")

# A tower: an octagonal prism standing on the ground...
tower = scene.add_geometry(
    "Tower", "cylinder", radius=1.5, depth=Expr("height"), segments=8, smooth=False
)
tower.add_modifier("transform", translation=Expr("(0, 0, height / 2)"))

# ...with window slits cut all around it (booleans need `aether3d[boolean]`).
slit = scene.add_geometry("Slit", "box", width=0.3, depth=1.2, height=1.0)
slit.add_modifier("transform", translation=Expr("(0, 1.5, height * 0.7)"))
slit.add_modifier(
    "array",
    count=Expr("windows"),
    relative_offset=(0, 0, 0),
    rotation=Expr("(0, 0, 360 / windows)"),
)
tower.add_modifier("boolean", operation="difference", operand={"geometry": "Slit"})

left = scene.add_object("Left Tower", geometry=tower)
right = scene.add_object("Right Tower", geometry=tower, position=(8, 0, 0))
assert left.mesh is right.mesh  # one evaluated mesh, shared

proc.set_parameter("windows", 6)  # both towers regenerate, nothing else does
scene.save("towers.aether")  # definitions are saved, not the evaluated mesh
scene.export("towers.glb")  # exports always use the current evaluation
```

[`examples/minecraft_castle.py`](../examples/minecraft_castle.py) builds a
complete castle this way. Its curtain walls carry battlements and an arched
gate, and its towers have arrow slits and crowns; there are also a keep, a
watchtower and shared bush instances. All of it is controlled by
`castle_width`, `wall_height`, `wall_thickness`, `tower_count`,
`tower_radius`, `gate_width`, `gate_height` and `detail_density`. The example
validates the castle, saves and reloads it, modifies it, exports and checks
it (Khronos validator, optional Blender), and undoes the changes.

## Concepts

| Term | Meaning |
| --- | --- |
| `scene.procedural` | The scene's `ProceduralSystem`: parameters, rules, geometry nodes, evaluation, cache and history. |
| Geometry node (`ProceduralGeometry`) | A reusable definition with a stable id and a unique name. It has its own parameters and rules, one **source**, and a **modifier stack**. |
| Source | `primitive` (a type and its parameters), `mesh` (a stored base `Mesh`), or `geometry` (another node's result, with parameter overrides, optionally taken *after* a given modifier). |
| Modifier (`Modifier`) | One stack entry. It has a stable id (UUID), a unique name within its stack, a type, validated parameters, and an `enabled` flag. |
| Procedural object | A `SceneObject` whose `geometry` is a node. `obj.mesh` is the node's evaluated `Mesh`. |

Evaluation turns a definition into a `PolyMesh`: an immutable polygon mesh
with per-corner UVs and normals, smooth flags and sharp edges, which flows
through the stack. The final `PolyMesh` is triangulated into an ordinary
`Mesh`, so inspection, glTF export and the CLI work unchanged.

Base definitions are never modified. A primitive's parameters, a stored base
mesh and every modifier's parameters stay exactly as written, whatever the
stack produces.

## Parameters, expressions and rules

**Parameters** are typed values:

| Type | Values | Constraints |
| --- | --- | --- |
| `float`, `int` | numbers (`int` rejects non-integral values) | `min`, `max` |
| `bool` | `true` / `false` | |
| `vec3` | three numbers | |
| `enum` | one of `choices` | `choices` |
| `string` | text | |

Scene parameters (`proc.add_parameter(...)`) are visible everywhere.
Geometry parameters (`geometry.add_parameter(...)`, or `parameters=` in
`add_geometry`) are local to a node and may shadow scene parameters. When the
node is referenced, they can be overridden. Each parameter can have a `unit`
and a `description`; agents can read both back with `info()`.

**Expressions** (`Expr("...")` in Python, `{"expr": "..."}` in JSON) compute a
value from parameters. Any parameter value, primitive parameter or modifier
parameter can be an expression, including a single component of a vector
(`Expr("(0, 0, height / 2)")`). The expression language is deliberately small
and safe, and is never Python `eval`:

- numbers, `true`/`false`, parameter names, `pi`;
- `+ - * / // % **` (numbers only), unary `-`/`+`/`not`;
- comparisons (chainable), `and`, `or`, `x if cond else y`;
- tuples `(a, b, c)`;
- functions `min max abs floor ceil round sqrt sin cos tan clamp int float`
  (angles in **degrees**).

Attribute access, indexing, strings, lambdas, keyword arguments and any other
names or calls are rejected (`INVALID_EXPRESSION`). Expressions are limited to
1,000 characters and 200 syntax nodes, exponents to |e| ≤ 64, and integer
results to 1,024 bits, so an expression cannot exhaust memory or time.

Resolved values are type- and range-checked **after** evaluation, so
`min`/`max` constrain derived values too. Parameters may depend on each other.
Cycles are rejected with `DEPENDENCY_CYCLE`, and the cycle is listed in
`details`.

**Rules** forbid invalid *combinations* of values:

```python
proc.add_rule("even_towers", "tower_count % 2 == 0", "tower_count must be even")
geometry.add_rule("arch_fits", "gate_width / 2 < gate_height")  # geometry scope
```

A rule is a boolean expression. A change that makes any rule false, including
through overrides, is rejected with `CONSTRAINT_VIOLATED`, the rule's message,
and the offending values in `details`. Primitives also check their own
combinations, for example a cone with both radii zero or a torus whose tube is
larger than its ring (`INVALID_COMBINATION`).

## Primitives

All primitives are centered at the origin, are watertight where they are
closed, wind outward, and carry per-corner UVs and analytic normals.
`aether3d procedural types` (or `ProceduralSystem.types()`) prints the full
schemas.

| Type | Parameters (default) | Notes |
| --- | --- | --- |
| `box` | `width`, `depth`, `height` (1) | 6 quads, flat |
| `grid` | `x_size`, `y_size` (2), `x_segments`, `y_segments` (1) | open, in XY facing +Z |
| `cylinder` | `radius` (1), `depth` (2), `segments` (32), `caps` (true), `smooth` (true) | n-gon caps with sharp rims |
| `cone` | `radius_bottom` (1), `radius_top` (0), `depth` (2), `segments` (32), `caps`, `smooth` | a 0 radius makes an apex; a frustum otherwise |
| `torus` | `major_radius` (1), `minor_radius` (0.25), `major_segments` (48), `minor_segments` (12) | all quads |
| `uv_sphere` | `radius` (1), `segments` (32), `rings` (16) | quads plus pole triangles |

## Modifiers

Modifiers never mutate their input. A modifier that cannot handle its input
raises a `ProceduralError` with a stable code instead of producing damaged
geometry. Results that are valid but noteworthy are reported as
*diagnostics* (see [codes](#error-and-diagnostic-codes)). `MAX_FACES`
(5,000,000) bounds what any modifier may produce (`PROCEDURAL_LIMIT`).

**Face groups** select per-face materials (see
[MATERIALS.md](MATERIALS.md#face-groups-several-materials-on-one-mesh)). They
come from a stored base mesh (`Mesh.groups`, or `geometry.edit_source()` with
`set_group`) and flow through every modifier: copies, mirrored halves,
solidify shells and rims, subdivided and beveled faces all keep the group of
the face they come from. Primitives have no groups (every face is in group 0).

### `transform`

`translation`, `rotation` (XYZ Euler, degrees), `scale` (non-zero), `pivot`.
Each point becomes `R·S·(p − pivot) + pivot + translation`. Stored normals
are transformed by the inverse transpose and renormalized. A mirroring
transform (negative determinant) re-winds every face so it stays
outward-facing.

### `mirror`

`axes` (`x`…`xyz`, applied in X, Y, Z order), `origin` (a point on every
plane), `merge` (true), `merge_threshold` (1e-4), `mirror_u`, `mirror_v`,
`bisect` (false), `bisect_flip` (false).

With `bisect`, the geometry is first cut at each plane with the editing
engine's bisect and only the positive side (with `bisect_flip`, the negative
side) is kept, so geometry crossing the plane no longer overlaps its copy
([MODELING.md](MODELING.md#procedural-creases-and-mirror-bisect)).

Each axis adds a reflected, re-wound copy of the current geometry. With
`merge`:

- Vertices within the threshold of the plane are snapped onto it and shared
  by both halves, so a half-model joins into one seamless mesh.
- Faces lying in the plane are not duplicated. If such a face is a cap
  between the halves (each of its edges also borders an off-plane face), it
  would end up *inside* the joined result, so it is removed and reported
  (`MIRROR_INTERIOR_FACES_REMOVED`). Other faces in the plane are kept once
  (`MIRROR_FACES_ON_PLANE`).
- Geometry on *both* sides of a plane is mirrored as requested but reported
  (`MIRROR_OVERLAP`), because the copy overlaps the original.

Without `merge`, the halves stay separate, and faces lying exactly in the
plane get coincident copies (`MIRROR_COINCIDENT_FACES`). Copied UVs are
unchanged unless `mirror_u`/`mirror_v` flip them (`u → 1 − u`).

### `array`

`count` (≥ 1), `relative_offset` (fraction of the bounding-box size per axis,
default `(1, 0, 0)`), `constant_offset`, `rotation`, `scale`, `pivot`,
`merge`, `merge_distance`.

Copy *k* is the input transformed by the step transform applied *k* times:
rotate and scale about `pivot`, then translate by
`relative_offset × size + constant_offset`. A rotation with a zero offset
makes radial arrays, such as the castle's tower crowns.

With `merge`, *open-boundary* vertices of different copies within
`merge_distance` are welded. Clusters are transitive and keep the lowest
vertex. Any face that collapses is removed and reported (`FACES_COLLAPSED`),
and `ARRAY_MERGED` reports how many vertices were welded.

### `solidify`

`thickness` (non-zero), `offset` (−1…1), `even_thickness`, `rim`.

With angle-weighted vertex normals `n`, the outer shell is
`p + n·t·(1 + offset)/2` and the inner shell (re-wound) is
`p − n·t·(1 − offset)/2`. So `offset = −1` keeps the original surface as the
outside, `+1` as the inside, and `0` centers it. `even_thickness` divides by
the mean cosine between the vertex normal and the adjacent face normals
(clamped to 4×) to keep walls uniform at corners. Every open boundary edge
gets a rim quad.

Rim faces copy the boundary UVs and therefore have zero UV area, which is
reported as `UV_DEGENERATE_FACES`. Input must be edge-manifold and
consistently wound; a vertex whose normal is undefined fails with
`UNDEFINED_NORMAL`. A closed input produces a hollow shell.

### `subdivision`

`levels` (0–6), `shade_smooth`, `crease` (0–10, default 0).

This is Catmull–Clark subdivision:

- **Face point:** the centroid of the face.
- **Interior edge point:** `(a + b + f₁ + f₂)/4`.
- **Interior vertex:** `(F + 2R + (n − 3)P)/n`.
- **Boundaries:** edge points are midpoints, and boundary vertices follow the
  cubic B-spline rule `(e₁ + e₂ + 6P)/8`. Corners, meaning vertices with
  other than two boundary edges, stay fixed.

Each step turns every n-gon into n quads: a cube becomes 24 quads at level 1,
with its corners at ±5/9. UVs are interpolated linearly within each face, so
UV seams are kept. Sharp edges are carried to the corresponding sub-edges as
shading information. With `crease` > 0 they are also semi-sharp creases
(DeRose, Kass and Truong 1998): sharp for `crease` levels (boundary rules),
blended for fractions; 0 keeps the smooth result of earlier versions bit for
bit. The input must be edge-manifold and consistently wound.

### `bevel`

`width`, `limit` (`all`, `angle`, `sharp`), `angle` (degrees; default 30),
`segments` (1-64, default 1), `profile` (0.01-0.99, default 0.5).

This is the mesh editing engine's bevel ([EDITING.md](EDITING.md)): a chamfer,
or with `segments` > 1 a rounded strip whose cross-section is a superellipse
`profile` ([MODELING.md](MODELING.md#rounded-bevels)), applied to the selected
edges with the same geometry, attribute and error rules. Error codes such as `BEVEL_WIDTH_TOO_LARGE` or
`BEVEL_UNSUPPORTED_VERTEX` pass through unchanged. `limit="angle"` selects
edges whose two faces meet at `angle` degrees or more; `sharp` selects edges
marked sharp, for example a cylinder's cap rims. If no edges match, the
modifier reports `BEVEL_NO_EDGES` and leaves the mesh unchanged.

### `boolean`

`operation` (`union`, `difference`, `intersection`), `operand`
(`{"geometry": id-or-name, "parameters": {overrides}}`), and
`operand_translation`/`operand_rotation`/`operand_scale`.

The operand is another geometry node, evaluated with optional parameter
overrides, so one cutter definition can make holes of several sizes. It is
not an object, so a geometry node stays shareable and independent of object
placement. The node reference is stored by id, so renaming the operand node
is safe.

The backend is **[Manifold](https://github.com/elalish/manifold)**
(`manifold3d`, Apache-2.0). It is an optional dependency: install it with
`pip install "aether3d[boolean]"`. Without it, evaluating a boolean fails
with `BOOLEAN_BACKEND_UNAVAILABLE`, while every other feature, including
loading projects that contain booleans, keeps working.

Manifold guarantees manifold output for valid input, and Aether3D validates
the input first. Both meshes must be closed, edge- and vertex-manifold,
consistently wound and of positive volume; otherwise the modifier fails with
`BOOLEAN_INPUT_NOT_SOLID`, and the topology report is in `details`.

- The result is triangulated.
- UVs are interpolated by the backend. If only one input has UVs, they are
  dropped and reported (`UVS_DROPPED`).
- Smooth/flat shading follows the face each triangle came from.
- Face groups follow it too: Manifold reports every output triangle's source
  face. If a backend did not, the groups would be dropped and reported
  (`FACE_GROUPS_DROPPED`).
- An empty result is reported (`BOOLEAN_EMPTY_RESULT`), not an error.

Operands whose pieces overlap *each other* (a self-intersecting operand)
are outside what Aether3D validates. Keep arrayed cutters apart, as the
castle does by sizing its crown merlons from the tower radius.

## Geometry references

```python
tall = scene.add_geometry("Tall Tower", source="Tower", overrides={"height": 12.0})
bare = scene.add_geometry("Bare Tower", source="Tower", after="Transform")
tall.add_modifier("boolean", operation="union", operand={"geometry": "Roof"})
```

A `geometry` source evaluates another node with parameter overrides, either
completely or only up to and including the modifier named by `after`. The
referencing node then applies its own stack. References and boolean operands
form a dependency graph:

- `proc.dependencies(node)` lists the nodes it reads, in evaluation order.
- `proc.dependents(node)` lists the nodes that read it.
- A cycle is rejected with `DEPENDENCY_CYCLE`.
- A node that is used by an object or referenced by another node cannot be
  removed (`GEOMETRY_IN_USE`).

## Evaluation, caching and performance

Evaluation is pure and **content-addressed**. Each step's cache key is a
SHA-256 over its fully resolved inputs:

- the primitive type and its resolved parameters, or the base mesh's content
  hash;
- the upstream key;
- the modifier type and its resolved parameters;
- the boolean operands' keys.

So:

- **Never stale.** Any change anywhere produces new keys downstream. Old
  results can never be returned for new inputs.
- **Minimal recomputation.** Unchanged stack prefixes, unchanged nodes and
  identical operands are reused, even across nodes. Changing one array count
  recomputes only that step and the steps after it. Setting a parameter back
  to an earlier value is served from the cache.
- **Sharing.** Objects using the same node, with the same resolved inputs,
  get the *same* `Mesh` object, which glTF export writes once.
- **Bounded memory.** The cache is a byte-bounded LRU
  (`proc.cache_limit_bytes`, default 512 MiB). `proc.cache_info` reports
  entries, bytes, hits, misses and the steps computed per kind;
  `proc.clear_cache()` empties it.

Measured with `python benchmarks/bench_procedural.py` (Python 3.14, NumPy
2.5, Linux x86-64; the best of 3 runs; peak memory from `tracemalloc`, which
does not include memory inside the Manifold library):

| Operation (UV sphere input; time / peak memory) | 512 faces | 8,192 faces | 51,200 faces |
| --- | --- | --- | --- |
| transform | 0.3 ms / 0.2 MiB | 1.6 ms / 4 MiB | 10 ms / 23 MiB |
| mirror (merge) | 0.7 ms / 0.6 MiB | 2.7 ms / 9 MiB | 20 ms / 57 MiB |
| array x10 | 1.5 ms / 2 MiB | 13 ms / 27 MiB | 81 ms / 167 MiB |
| array x10, merge (grid input) | 15 ms / 4 MiB | 184 ms / 66 MiB | 1.1 s / 406 MiB |
| solidify (grid input) | 1.4 ms / 0.7 MiB | 19 ms / 10 MiB | 131 ms / 62 MiB |
| subdivision L1 | 1.3 ms / 0.6 MiB | 22 ms / 10 MiB | 168 ms / 65 MiB |
| subdivision L2 | 6.0 ms / 3 MiB | 113 ms / 54 MiB | — |
| bevel (every edge) | 181 ms / 9 MiB | 3.1 s / 152 MiB | 16.8 s / 811 MiB |
| boolean difference | 13 ms / 1 MiB | 184 ms / 17 MiB | 1.3 s / 104 MiB |
| final triangulation (`to_mesh`) | 1.3 ms / 0.6 MiB | 19 ms / 10 MiB | 142 ms / 64 MiB |

Castle demo (6,754 triangles, 15 geometry nodes): cold build and evaluation 112 ms
(7 MiB peak); re-evaluating everything when nothing changed 2 ms (all cache hits);
changing `castle_width`, which affects every node, 74 ms. Changing a value back to an
earlier one is served from the cache.

`bevel` is the slowest modifier because it runs the editing engine's
per-element (Python) bevel. Every other modifier is vectorized NumPy.

## Transactions, undo and errors

Every mutating call is a **transaction**:

1. The change is applied.
2. References, schemas, rules, parameter and dependency cycles are checked.
3. Every geometry node is evaluated, which costs little when results are
   cached.

If any step fails, every definition, every object's geometry assignment and
every handle (`Modifier`/`ProceduralGeometry` objects you hold) is restored
exactly, and a `ProceduralError` is raised. The error has a stable `code`, a
`location` (for example `geometry[Tower].modifiers[Row].params.count`), an
optional `hint` and `details`. Problems therefore surface at the change that
caused them, not later at export time.

```python
with proc.batch():  # several changes, validated once, one undo step
    proc.set_parameter("castle_width", 32.0)
    proc.set_parameter("tower_count", 8)
proc.undo()  # back to the previous castle; proc.redo() re-applies it
```

`proc.revision` increases with every successful change. Undo and redo cover
procedural definitions and object assignments, up to `max_history` (256)
steps:

- `add_object(geometry=)`, `set_geometry()` and `bake()` are steps in this
  history.
- Objects that were static both before and after a step are never touched,
  so a static mesh assigned in between is kept.
- A step that would leave an object referencing a node that no longer exists
  is refused with `HISTORY_CONFLICT`, and nothing changes.

The editing engine's own undo is separate and lives inside each editing
session.

Since 0.9.0 every procedural change is also a step of the scene's history,
`scene.history` ([HISTORY.md](HISTORY.md)), which covers all kinds of changes.
`proc.undo()` and `proc.redo()` remain the procedural-only history above, and
each call is itself a step of the scene's history. `proc.batch(label)` names its
step.

## Objects: sharing, baking and editing the source

- `scene.add_object(name, geometry=node)` creates a procedural object, and
  `scene.add_procedural(name, "box", **params)` creates a primitive node and
  its object in one atomic step. `obj.set_geometry(node_or_None)` reassigns
  an existing object.
- `obj.mesh` is the evaluated mesh. It is read-only for procedural objects:
  assigning one raises `PROCEDURAL_OBJECT`, with a hint.
- `obj.edit_mesh()` on a procedural object raises `PROCEDURAL_OBJECT`, because
  a destructive edit would be lost at the next regeneration. Instead:
  - change the definition;
  - `obj.bake()`: replace the procedural geometry with a static copy of the
    evaluated mesh (the node stays for other users, and the bake is
    undoable); or
  - edit the *base mesh* of a node with `node.edit_source()`. That works like
    an editing session, and on `commit()` it replaces the stored base mesh,
    so every user regenerates with all modifiers re-applied. Primitive
    sources can first be converted with `node.convert_source_to_mesh()`.

## Persistence

Projects are saved as `.aether` **format version 2**; see
[FILE_FORMAT.md](FILE_FORMAT.md).

- `scene.parameters`, `scene.rules` and `scene.geometry` hold the
  definitions. Expressions are stored as `{"expr": "..."}`.
- Each object has a `geometry` reference (or `null`).
- Procedural objects store **no** evaluated mesh, so a saved file can never
  contain stale geometry. Stored base meshes go in the shared `meshes` table.
- Loading validates the complete structure (types, schemas, references,
  rules, cycles) and reports problems as `ProjectFormatError` with a location.
  Evaluation is lazy, so a project that uses booleans loads even without
  `manifold3d`.

Version 1 files (Aether3D 0.1–0.3) load unchanged: they are migrated to
empty procedural tables, and the next save writes version 2.

## JSON scripts

Agents and tools can change projects declaratively. The script below is
complete, and the test suite runs it:

```json
{
  "format": "aether3d.procedural-script",
  "version": 1,
  "operations": [
    {"op": "add_parameter", "name": "width", "value": 12.0, "min": 4, "max": 60, "unit": "m"},
    {"op": "add_parameter", "name": "posts", "value": 4, "min": 2, "max": 20},
    {"op": "add_rule", "name": "spacing", "expr": "width / (posts - 1) >= 1",
     "message": "posts must be at least 1 m apart"},
    {"op": "add_geometry", "name": "Post", "primitive": "box",
     "params": {"width": 0.3, "depth": 0.3, "height": 2}},
    {"op": "add_modifier", "geometry": "Post", "type": "array", "name": "Row",
     "params": {"count": {"expr": "posts"}, "relative_offset": [0, 0, 0],
                "constant_offset": {"expr": "(width / (posts - 1), 0, 0)"}}},
    {"op": "add_modifier", "geometry": "Post", "type": "bevel",
     "params": {"width": 0.03, "limit": "all"}},
    {"op": "create_object", "name": "Fence", "geometry": "Post"},
    {"op": "set_parameter", "name": "posts", "value": 7},
    {"op": "validate", "require_ok": true}
  ]
}
```

Operations:

| Operation | Keys |
| --- | --- |
| `add_parameter` | `name`, `value`; optional `type`, `min`, `max`, `choices`, `unit`, `description`, `geometry` |
| `set_parameter`, `remove_parameter` | `name` (+ `value`); optional `geometry` |
| `add_rule`, `remove_rule` | `name` (+ `expr`, `message`); optional `geometry` |
| `add_geometry` | `name`, and exactly one of `primitive` + `params`, `source` + `overrides` / `after`, or `mesh_from_object`; optional `define` (a list of parameter definitions) |
| `remove_geometry`, `rename_geometry`, `set_source`, `convert_source_to_mesh` | `geometry` (+ `name` / `params`) |
| `add_modifier` | `geometry`, `type`; optional `name`, `enabled`, `index`, `params` |
| `set_modifier` | `geometry`, `modifier` (id, name or index); optional `params`, `enabled`, `name` |
| `remove_modifier`, `move_modifier` | `geometry`, `modifier` (+ `index`) |
| `create_object` | `name`, `geometry`; optional `parent`, `position`, `rotation_euler`, `scale` |
| `assign`, `bake` | `object` (+ `geometry`, which may be `null`) |
| `evaluate`, `validate`, `info` | read-only; `validate` takes `require_ok` |

Scripts are data. Nothing in them is executed as code, and expressions use
the restricted language above. The whole script is **one transaction**: each
operation is validated as it runs, so the error names the failing operation
(`operations[3]: geometry[Post].modifiers[Row].params.count`). On any failure,
every definition is restored and the objects the script created are removed.
Unknown keys, operations, formats and versions are rejected.

## Command line

```console
$ aether3d procedural types --json                  # every primitive/modifier schema
$ aether3d procedural run castle.aether build.json  # apply a JSON script (atomic)
$ aether3d procedural inspect castle.aether --evaluate
$ aether3d procedural set castle.aether tower_count=8 castle_width=32
$ aether3d procedural set castle.aether radius=2 --geometry Tower
$ aether3d procedural validate castle.aether        # exit 3 if any geometry fails
$ aether3d procedural bake castle.aether --object "Towers"   # or --all
```

`run`, `set` and `bake` accept `--output PATH` (instead of updating the
project) and `--dry-run`. Values in `set` are parsed as JSON (`3`, `2.5`,
`true`, `[1,2,3]`, `{"expr": "width / 2"}`), or taken as plain strings for
enum values. Every command supports `--json`.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | failure, with nothing written |
| `2` | usage error |
| `3` | `procedural validate` found errors |

Other commands (`geometry`, `export`, `inspect`, …) see procedural objects as
their evaluated meshes. If evaluation fails, they exit with `1` and a
`ProceduralError` whose location names the failing modifier.

## Introspection for AI agents

- `ProceduralSystem.types()` gives every primitive and modifier schema:
  parameter types, defaults, ranges, choices and descriptions. It also lists
  the expression functions.
- `proc.info(evaluate=True)` gives every parameter (raw value and resolved
  value), every rule and every node. For each node it includes the source,
  each modifier's raw and resolved parameters, dependencies, dependents,
  users, and the evaluated statistics and diagnostics. It is JSON-compatible.
- `proc.validate()` evaluates every node and reports per-node polygon
  topology, mesh inspection and modifier diagnostics. It never raises for a
  geometry problem: a failing node is reported with its error.
- `proc.parameter_dependencies()` shows which parameters each expression
  reads.

Recommended workflow:

1. Read `types()`.
2. Make changes in one `batch()` or one script.
3. Read the `ProceduralError` code and location on failure, and `validate()`
   on success.
4. Use ids for durable references. Names are unique and convenient but can be
   renamed.

## Error and diagnostic codes

Errors (`ProceduralError.code`, or `ScriptError.code` in scripts):

| Code | Meaning |
| --- | --- |
| `INVALID_PARAMETER`, `PARAMETER_OUT_OF_RANGE`, `UNKNOWN_PARAMETER`, `MISSING_PARAMETER`, `DUPLICATE_PARAMETER` | Bad parameter value, name or schema |
| `INVALID_EXPRESSION`, `EXPRESSION_ERROR` | Disallowed syntax, or an evaluation failure (division by zero, overflow, …) |
| `CONSTRAINT_VIOLATED`, `INVALID_RULE`, `DUPLICATE_RULE`, `UNKNOWN_RULE` | Rules |
| `INVALID_COMBINATION` | A primitive's parameters contradict each other |
| `DEPENDENCY_CYCLE` | Parameter expressions or geometry references form a cycle |
| `UNKNOWN_GEOMETRY`, `UNKNOWN_MODIFIER`, `UNKNOWN_PRIMITIVE`, `UNKNOWN_OBJECT`, `GEOMETRY_IN_USE`, `DUPLICATE_NAME`, `DUPLICATE_ID`, `INVALID_NAME`, `INVALID_INDEX`, `INVALID_SOURCE` | Definition structure |
| `DEGENERATE_TRANSFORM` | Zero scale |
| `UNSUPPORTED_TOPOLOGY`, `INCONSISTENT_WINDING`, `UNDEFINED_NORMAL` | The input is unsuitable for solidify or subdivision |
| `BOOLEAN_INPUT_NOT_SOLID`, `BOOLEAN_BACKEND_UNAVAILABLE`, `BOOLEAN_FAILED` | Booleans |
| `BEVEL_*`, `NOT_MANIFOLD`, `DEGENERATE_INPUT` | Passed through from the editing engine's bevel |
| `PROCEDURAL_LIMIT`, `NON_FINITE_RESULT`, `MODIFIER_FAILED`, `TRIANGULATION_FAILED` | Resource limits and numerical failures |
| `PROCEDURAL_OBJECT`, `SOURCE_NOT_EDITABLE` | An operation that does not apply to procedural objects or sources |
| `NOTHING_TO_UNDO`, `NOTHING_TO_REDO`, `HISTORY_CONFLICT` | History |

Diagnostics (results that are valid but noteworthy):

| Code | Severity | Meaning |
| --- | --- | --- |
| `MIRROR_OVERLAP` | warning | Geometry spans a mirror plane; the copy overlaps it |
| `MIRROR_FACES_ON_PLANE` | warning | Faces in the plane were kept once, not duplicated |
| `MIRROR_COINCIDENT_FACES` | warning | `merge` is off: faces in the plane coincide with their copies |
| `MIRROR_INTERIOR_FACES_REMOVED` | info | Cap faces that would lie inside the merged result were removed |
| `ARRAY_MERGED` | info | Number of vertices welded between copies |
| `FACES_COLLAPSED` | warning | Faces collapsed by welding and were removed |
| `SOLIDIFY_OPEN` | info | `rim` is off, so the result has open boundaries |
| `UV_DEGENERATE_FACES` | warning | Some faces have zero UV area (solidify rims) |
| `UVS_DROPPED` | warning | Only one boolean input had UVs |
| `BOOLEAN_EMPTY_RESULT` | warning | The boolean result is empty |
| `FACE_GROUPS_DROPPED` | warning | The boolean backend did not report source faces, so face groups were lost |
| `BEVEL_NO_EDGES` | info | No edge matched the bevel limit |

`proc.validate()` additionally reports polygon topology codes: those of
[EDITING.md](EDITING.md) (`BOUNDARY_EDGES`, `NON_MANIFOLD_EDGES`,
`INCONSISTENT_WINDING`, …) plus the codes of [INSPECTION.md](INSPECTION.md).

## Limitations

- **Self-intersection is not detected.** Overlapping parts of one mesh are
  not reported, for example an array with overlapping copies, or a mirror of
  geometry that spans the plane, which *is* reported as `MIRROR_OVERLAP`.
  Boolean operands should not overlap themselves.
- **Mirror without `bisect`** mirrors geometry crossing a plane whole
  (`MIRROR_OVERLAP`); set `bisect` to cut it first.
- **Subdivision** creases apply one sharpness to every edge marked sharp
  (no per-edge weights), and UVs are subdivided linearly per face rather than
  smoothed.
- **Bevel** follows the editing engine's topology rules (manifold edges; each
  vertex one fan, possibly open). It is the slowest modifier on large meshes
  (see the benchmarks).
- **Boolean** results are triangulated (polygons are not rebuilt). Inputs
  must be closed solids, and the operation needs `manifold3d`.
- **Instancing:** evaluated meshes are shared per node and resolved inputs.
  Per-object parameter overrides are made by creating a referencing node
  (`source=` with `overrides=`), not on the object itself.
- **Precision:** evaluation is float64. Evaluated `Mesh` objects store float32
  positions, like all Aether3D meshes and glTF.
- Procedural modifiers cover geometry only. Materials, textures, animation
  and rendering are outside this stage.
