# Scene plans

A **scene plan** is one JSON document (`aether3d.scene-plan`, version 1) that builds or
changes a scene: meshes, materials, procedural geometry, reusable structures (prefabs),
objects placed by patterns, edits of existing objects, the world, the active camera, and
spatial questions to answer afterwards. A plan is:

- **checked as a whole first.** Every problem comes back at once, each with a stable
  code, a JSON pointer, the operation index and a hint. Nothing changes until the plan is
  known to be valid.
- **applied as one undo step, all or nothing.** A failure (or a timeout, or a
  cancellation) rolls the scene back to exactly where it was.
- **data, not code.** No Python is run. Expressions use the procedural engine's safe
  evaluator: arithmetic, comparisons, `and`/`or`/`not`, `a if c else b`, the constant
  `pi` and the functions `min`, `max`, `abs`, `floor`, `ceil`, `round`, `sqrt`, `sin`,
  `cos`, `tan` (degrees), `clamp`, `int` and `float` -- nothing else (no attributes, no
  other calls, no `eval`).
- **fast for many objects.** A grid of 10,000 blocks is one call that creates the objects
  in bulk ([performance](#performance)): an agent writes one small document instead of
  thousands of statements.

The same plan runs from Python (`scene.execute_plan(plan)`), from the command line
(`aether3d execute-plan`), and from AI agents over MCP (the `execute_scene_plan` tool of
[`aether3d mcp serve`](MCP.md)). The JSON Schema is `aether3d plan schema`
(`aether3d.plans.plan_schema()`, the MCP resource `aether3d://schema/scene-plan`); a short
guide written for agents is `aether3d://guide` (`src/aether3d/plans/guide.md`).

## A first plan

```json
{
  "format": "aether3d.scene-plan",
  "version": 1,
  "label": "Courtyard",
  "materials": {"stone": {"preset": "stone"}, "red": {"base_color": "#a03020"}},
  "meshes": {"block": {"primitive": "cube", "size": 1}},
  "objects": {
    "ground": {"mesh": {"primitive": "plane", "size": 40}, "color": "#4f7f3a"},
    "wall": {
      "mesh": "@block", "material": "@stone", "position": [-5, 0, 0.5],
      "grid": {"count": [11, 1, 3]},
      "where": {"expr": "not (x >= 4 and x <= 6 and z <= 1)"}
    },
    "pillars": {
      "mesh": {"primitive": "cylinder", "radius": 0.3, "depth": 3},
      "circle": {"count": 8, "radius": 6, "z": 1.5}, "material": "@red"
    },
    "sun": {"light": {"type": "directional", "intensity": 2}, "direction": [-0.4, 0.5, -1]},
    "cam": {"camera": {"fov": 45}, "position": [14, -16, 9], "look_at": [0, 0, 1], "active": true}
  },
  "queries": [{"query": "overlaps", "objects": "@pillars", "against": "@wall"}]
}
```

```python
from aether3d import Scene

scene = Scene("Courtyard")
result = scene.execute_plan(plan)  # a dict, or a JSON string
result.status  # "applied"
result.counts["objects"]  # 38: ground, 27 wall blocks (33 minus a gap), 8 pillars, sun, camera
result.queries[0]["result"]  # [] -- no pillar touches the wall
scene.history.undo()  # the whole plan, in one step
```

```bash
aether3d new yard.aether
aether3d execute-plan yard.aether courtyard.json      # checked, applied, saved
aether3d plan validate courtyard.json                 # check only (exit status 3 on problems)
```

Conventions are the engine's: +Z up, right-handed, meters; objects face -Y (cameras look
along -Y); `rotation_euler` in degrees (XYZ); colors are sRGB `"#rrggbb"` or `[r, g, b]` in
0..1. Primitives are centered on their origin: a cube of size 1 stands on the ground at
`z = 0.5`.

## The document

| Key | Meaning |
| --- | --- |
| `format`, `version` | `"aether3d.scene-plan"`, `1` (required) |
| `label`, `description` | the undo step's label; free text |
| `expected_revision` | refuse to apply (`REVISION_MISMATCH`) unless `scene.revision` is this |
| `idempotency_key` | apply once: a resent plan reports `already_applied` ([retries](#retries-and-idempotency)) |
| `options` | [options](#options) |
| `parameters` | named values for expressions (numbers, booleans, 3-vectors, expressions of other parameters) |
| `procedural` | scene parameters of the procedural engine to add (`parameters`) or set (`set`) |
| `meshes`, `textures`, `materials`, `geometries`, `prefabs` | resources, by symbol |
| `objects` | objects to create, by symbol |
| `operations` | edits, in order, after the objects are made |
| `world`, `render_settings` | as `scene.world` / `scene.render_settings` take them |
| `active_camera` | a camera to make active |
| `queries` | spatial questions answered after the plan is applied |

Sections run in this order: parameters, resources, objects, operations, world and render
settings, the active camera, queries. Within `objects`, parents are made before their
children; within `operations`, entries run in the order given.

## Symbols and references

Every key of `meshes`, `textures`, `materials`, `geometries`, `prefabs` and `objects` is a
**symbol**: ASCII letters, digits and `_ - .`, at most 64 characters, starting with a
letter, digit or `_`, and unique across all those sections.

A plan names what it makes with `"@symbol"`:

| Reference | Names |
| --- | --- |
| `"@towers"` | every instance of the object `towers` |
| `"@towers[2]"`, `"@towers[-1]"` | one instance (negative indexes count from the end) |
| `"@towers/roof"` | the child `roof` of every instance |
| `"@towers[2]/roof"` | the child `roof` of one instance |
| `"@stone"` | the material (or mesh, texture, geometry, prefab) `stone` |

Instance indexes count the instances made: with a [`where`](#where-skipping-instances)
filter, `"@wall[3]"` is the fourth block created. A prefab is a template, not an object:
its copies are reached through the objects that use it (`"@street/roof"`), and a
reference to a prefab where an object is wanted is an error that says so.

What exists in the scene already is named without `@`: by id or name (`"Ground"`), or as
`{"id": "..."}` / `{"name": "..."}` when a name could be mistaken for something else.
Plan symbols exist only inside their plan; the next plan names the same objects by name or
id (every result lists the ids of what was made).

All references are resolved before anything changes. `UNKNOWN_REFERENCE` (with a "did you
mean" hint), `INVALID_REFERENCE` (the wrong kind of thing) and `AMBIGUOUS_REFERENCE` (many
objects where one is needed) point at the reference.

## Resources

**Meshes** are shared: every object that uses `"@block"` uses the same mesh, which the GPU
rasterizer draws in one instanced draw call per mesh. A mesh is one of:

- `{"primitive": ..., ...}`: `cube` (`size`), `box` (`size: [x, y, z]`, or `width`, `depth`,
  `height`), `sphere` and `uv_sphere` (`radius`, `segments`, `rings`), `plane` (`size`),
  `cylinder` (`radius`, `depth`, `segments`, `caps`), `cone` (`radius_bottom`,
  `radius_top`, `depth`, `segments`, `caps`), `torus` (`major_radius`, `minor_radius`,
  `major_segments`, `minor_segments`), `grid` (`x_size`, `y_size`, `x_segments`,
  `y_segments`); `smooth: false` gives flat shading;
- `{"data": {"positions": [...], "triangles": [...], "normals", "uvs", "groups"}}`;
- `{"from_object": "Existing object"}`: that object's mesh, shared.

Inline meshes on objects (`"mesh": {"primitive": "cube"}`) are shared too: identical
definitions in a plan make one mesh.

**Materials** take a `preset` (`stone`, `brick`, `wood`, `metal`, `grass`, `fabric`, with
the preset's options such as `color` and `scale`) or the material parameters of
[MATERIALS.md](MATERIALS.md) (`base_color`, `metallic`, `roughness`, `emission`, ...), a
`model`, and `textures` by slot (`{"base_color": {"texture": "@bricks"}}`, `{"image":
"..."}` or `{"procedural": ...}`). **Textures** are images (PNG or JPEG files inside the
project folder; paths that leave it are refused with `UNSAFE_PATH`, unless Python passes
`trusted_paths=True`) or procedural node graphs.

**Geometries** are procedural geometry nodes ([PROCEDURAL.md](PROCEDURAL.md)): a
`primitive` with `params`, or a `source` geometry with `overrides`, or a `mesh`, plus
`parameters` and `modifiers` (`{"type": "array", "params": {...}}`, booleans, bevels,
...). Each node is evaluated once, after each modifier, so a failing modifier is reported
at its own pointer. Objects with `"geometry": "@tower"` stay procedural: changing a
parameter later changes them all.

**Prefabs** are object definitions without placement, reused by objects
(`"prefab": "@house"`), including by other prefabs:

```json
"prefabs": {
  "house": {"tags": ["house"], "children": {
    "walls": {"mesh": "@box", "material": "@plaster", "position": [0, 0, 1.5]},
    "roof": {"mesh": "@roof", "material": "@tiles", "position": [0, 0, 3.5]}}}
},
"objects": {"street": {"prefab": "@house", "array": {"count": 12, "offset": [8, 0, 0]}}}
```

## Objects

An object definition has at most one **kind**: `mesh`, `geometry` (a procedural node),
`prefab`, `camera` (`{"projection", "fov", "fov_axis", "ortho_scale", "near", "far",
"aspect_ratio"}`) or `light` (`{"type": "directional" | "point" | "ambient", "color",
"intensity", "range", "falloff", "radius", "angle"}`). With none, the object is an empty
group.

| Key | Meaning |
| --- | --- |
| `name` | the object's name, or a [name template](#names) |
| `id` | an explicit id (otherwise new, or derived from the idempotency key) |
| `parent` | top-level objects: an object of the plan or of the scene |
| `position`, `rotation_euler`, `rotation` (quaternion), `scale` | placement (local to the parent) |
| `look_at` (a point or an object), `look_along`, `direction`, `up` | aiming (cameras, lights, anything) |
| `material`, `group_materials`, `color`, `specular`, `shininess` | appearance |
| `visible`, `tags`, `metadata` | as on scene objects |
| `active` | cameras: make it the scene's camera |
| `frame` | cameras: `true` or `{"targets", "direction", "margin"}`: frame the targets |
| `children` | objects made under every instance, by symbol |
| one pattern | [patterns](#patterns) |
| `vary`, `where` | [per-instance variation](#vary-variation-per-instance), [filters](#where-skipping-instances) |

Numbers, vectors and colors can be expressions: `{"expr": "spacing * 2"}`, and
`[{"expr": "i % 10"}, 0, 0.5]`.

### Names

Without a `name`, an object is called by its symbol; instances of a pattern are
`"symbol i"`, and children `"Parent/child"`. Templates fill in `{i}` (instance index),
`{i:3}` (zero-padded), `{n}` (instance count), `{x}`, `{y}`, `{z}` (grid indexes),
`{parent}` and `{symbol}`: `"name": "Block {x}-{y}"`.

Names in the scene are unique. A name that is taken is an error (`NAME_CONFLICT`) unless
`"options": {"on_name_conflict": "rename"}`, which gives it the next free `.NNN` suffix (and
a `NAMES_CHANGED` warning).

## Patterns

One pattern per object turns its definition into many instances:

| Pattern | Makes |
| --- | --- |
| `"grid": {"count": [nx, ny, nz], "spacing": s or [sx, sy, sz], "center": false}` | a box of instances, x fastest |
| `"array": {"count": n, "offset": [dx, dy, dz], "rotation_step": [..], "scale_step": f}` | a row, each step from the last |
| `"circle": {"count": n, "radius": r, "z": h, "start_angle": a, "end_angle": b, "orient": "outward"}` | a ring around +Z; `orient`: `outward`, `inward`, `tangent` or none |
| `"line": {"from": [..], "to": [..], "count": n}` | evenly spaced, both ends included |
| `"scatter": {"count": n, "box": {"min": [x, y], "max": [x, y]}, "seed": 1, "min_distance": d}` | random but reproducible (SplitMix64); or `"circle": {"center": [x, y], "radius": r}`; `z`, `allow_fewer`, `max_attempts` |
| `"instances": [{"position": .., "rotation_euler": .., "scale": .., "name": .., "material": ..}, ...]` or `{"positions": [[..], ...]}` | explicit instances |
| `"repeat": n` | `n` instances, placed with expressions over `i` |

The object's own `position`, `rotation_euler` and `scale` apply to every instance (added,
added and multiplied). Counts can be expressions of plan parameters. A `scatter` that
cannot place `count` points `min_distance` apart (with `max_attempts`, default 30, tries
per point) fails with `PATTERN_UNSATISFIABLE`, or, with `allow_fewer`, makes fewer and
warns `SCATTER_FEWER`.

### Instance variables

Expressions in a patterned object see `i` (0..n-1), `n`, `u` (`i / (n - 1)`, 0..1), `x`,
`y`, `z` (grid indexes) and `angle` (circle), next to the plan's `parameters`. Math
functions take degrees: `{"expr": "sin(angle) * 2"}`.

### `vary`: variation per instance

`vary` sets fields per instance: `offset`, `rotation_euler`, `scale`, `color`, `specular`,
`shininess`, `material`, `mesh`, `geometry`, `visible`, `tags`. Each takes a generator:

| Generator | Value of instance `i` |
| --- | --- |
| `{"cycle": [a, b, c]}` | `a, b, c, a, b, ...` |
| `{"choose": [a, b], "weights": [3, 1], "seed": 7}` | a random pick (reproducible) |
| `{"uniform": [low, high], "seed": 2}` | uniform between (numbers or vectors) |
| `{"linear": [start, end]}` | interpolated over the instances |
| `{"values": [one per instance]}` | given |
| `{"expr": "..."}` | an expression of the instance variables |

```json
"crates": {"mesh": "@crate", "scatter": {"count": 40, "box": {"min": [-8, -8], "max": [8, 8]}, "seed": 3},
           "vary": {"rotation_euler": {"uniform": [[0, 0, 0], [0, 0, 360]], "seed": 4},
                    "material": {"choose": ["@oak", "@pine"], "seed": 5}}}
```

### `where`: skipping instances

`"where": {"expr": "..."}` keeps only the instances for which the expression is true -- a
doorway in a wall of blocks, a gate in a fence, every third lamp:

```json
"front": {"mesh": "@block", "grid": {"count": [10, 1, 6]},
          "where": {"expr": "not (x >= 4 and x <= 5 and z <= 2)"}}
```

Names keep the pattern's indexes (`"front 23"` is still grid cell 23), `vary` generators
and expressions still see the pattern's `i` and `n`, and skipped instances make no
children. A static check counts patterns before filtering, so its estimate is an upper
bound (`estimate.objects_at_most`).

## Operations

Operations change objects after the plan's objects are made, in order:

```json
"operations": [
  {"op": "update", "select": {"tags": ["lamp"], "expect": 8}, "set": {"color": "#2b4a3a"}},
  {"op": "update", "select": "Tree 1", "set": {"position": [-8, -4, 0]}},
  {"op": "delete", "select": ["Tree 3", "Tree 7"], "recursive": true},
  {"op": "set_parameters", "values": {"towers": 6}},
  {"op": "frame_camera", "camera": "@cam", "targets": "@houses", "direction": [-1, 1, -0.6]},
  {"op": "set_active_camera", "camera": "@cam"}
]
```

- **`update`** `set` takes `name` (one object), `position`, `rotation_euler`, `rotation`,
  `scale`, `translate`, `scale_by`, `color`, `specular`, `shininess`, `material` (or
  `null`), `visible`, `tags`, `add_tags`, `remove_tags`, `metadata` (entries to set;
  `null` removes one), `parent` (or `null`; with `keep_world_transform`), `look_at`,
  `look_along`, `up`.
- **`delete`** removes the selection (`recursive`: with descendants, the default).
- **`set_parameters`** sets procedural parameters (of the scene, or of one `geometry`).
- **`frame_camera`** frames `targets` with a camera (optionally turned along `direction`).
- **`set_active_camera`** makes a camera the scene's camera.
- **`cad`** (0.15.0) runs a CAD script (`"script": {"format": "aether3d.cad-script",
  ...}`, [CAD.md](CAD.md#cad-scripts)) inside the plan's step: parametric sketches,
  parts, buildings and drawings. Its structure is checked with the rest of the plan
  (problems point into it: `/operations/0/script/operations/2/op`); a failure rolls the
  whole plan back. A CAD item the script shows (`{"op": "show", "item": ...}`) is an
  object later operations can select by name.

### Selectors

A selection is a reference (`"@towers"`, `"Ground"`), a list of them, or an object:

| Key | Selects |
| --- | --- |
| `refs`, `ids`, `names` | these objects (an unknown one is an error) |
| `tags` (all), `any_tags` | by tags |
| `name_glob` | by name pattern: `"Tower */roof"` |
| `kind` | `mesh`, `procedural`, `camera`, `light`, `empty` |
| `under` | descendants of an object |
| `visible`, `material` | by visibility, by material |
| `created` | only objects this plan made |
| `expect` | the count it must match: `8` or `{"min": 1, "max": 20}` |

Filters combine (all must hold); with `refs`, `ids` or `names`, they narrow those
objects. Objects come in scene order. A selection by filters that **deletes** must say how
many objects it means (`expect`): a delete that would remove more than intended fails with
`SELECTION_MISMATCH` instead. The same selectors work in queries and in the MCP tools
`objects_list` and `spatial_query`.

## Queries

`queries` are spatial questions ([SPATIAL.md](SPATIAL.md)), answered right after the plan
is applied, in the same call: `describe`, `bounds`, `find`, `raycast`, `raycast_all`,
`closest_point`, `nearest`, `contains`, `distance`, `overlaps`, `region`, `camera_ray`,
`pick`, `project`, `in_view`, `line_of_sight`, `visible_from`, `relation`. Plan references
(`"@pillars"`) become object ids, and object lists (`objects`, `against`, `exclude`) also
take selectors:

```json
"queries": [
  {"query": "overlaps", "objects": {"name_glob": "Tree */*"}, "against": {"under": "keep"}},
  {"query": "bounds", "object": "Tower NE", "children": true},
  {"query": "line_of_sight", "start": [0, -30, 2], "end": [0, -8, 2]}
]
```

Each answer is in `result.queries` (`{"index", "query", "result"}`); a query that fails
gives `{"index", "query", "error"}` and does not fail the plan. References in queries are
checked before anything is built.

## Options

| Option | Values |
| --- | --- |
| `on_name_conflict` | `"error"` (default) or `"rename"` |
| `tags` | tags added to every object the plan makes |
| `result_ids` | `"auto"` (default: list the ids of symbols with at most 20 objects), `"all"`, `"none"` |
| `chunk_size` | apply in [chunks](#large-plans-chunks) of this many objects (at least 100; needs an `idempotency_key`) |
| `on_chunk_failure` | `"keep"` (default: completed chunks stay) or `"rollback"` |
| `timeout` | seconds; past it, the plan rolls back with `TIMEOUT` |

## Modes and results

`scene.execute_plan(plan, mode=...)`:

| Mode | Does |
| --- | --- |
| `"execute"` (default) | check, apply as one undo step, answer the queries |
| `"dry_run"` | check, apply, report, then roll back exactly: the scene and its history are unchanged |
| `"validate"` | check against the scene and estimate, without applying |

`validate_plan(plan)` checks a plan without a scene (structure, types, symbols, references
between symbols, limits) and estimates counts; references to existing objects are checked
when a scene is given (`validate_plan(plan, scene)`, or `mode="validate"`). Other options:
`limits` ([limits](#limits)), `timeout`, `progress` (a callable `(done, total, message)`),
`cancel` (a `threading.Event`), `base_dir` (for relative texture images).

The result, `PlanResult` (`to_dict()` gives JSON):

| Field | Meaning |
| --- | --- |
| `ok`, `mode`, `status` | `applied`, `dry_run`, `validated`, `already_applied` |
| `label`, `fingerprint`, `idempotency_key` | the plan's label, a hash of its content, its key |
| `revision` | `{"before", "after"}` |
| `step` | the undo step: `{"index", "label", "changes", "bytes"}` |
| `counts` | objects (`by_kind`), meshes, materials, textures, geometries, operations, objects sharing a mesh |
| `symbols` | per symbol: its count and (see `result_ids`) the ids made, grouped (`houses/walls`) |
| `operations` | per operation: `{"op", "index", "matched", ...}` |
| `queries` | the answers |
| `warnings`, `issues` | non-fatal notes; the problems (for a check) |
| `estimate` | for checks: what the plan would make. Without a scene: objects (an upper bound when `objects_at_most`), per symbol, meshes, materials, ...; against a scene: exact objects, triangles, new mesh bytes, memory, the scene's size afterwards, chunks |
| `chunks` | for chunked runs: totals, completed, skipped, the checkpoint, the steps |
| `timings` | seconds per stage: `parse`, `bind`, `resources`, `objects`, `operations`, `apply`, `queries`, `total` |

## Errors

A plan that fails changes nothing and raises `PlanError` (an `AetherError`):

```python
try:
    scene.execute_plan(plan)
except PlanError as error:
    error.code  # "UNKNOWN_REFERENCE"
    error.pointer  # "/objects/bench/mesh"
    error.operation  # the index in "operations", or None
    error.hint  # 'define it, or name an existing object without "@"'
    error.issues  # every problem found, as dicts with the same fields
```

Stable codes:

| Code | Meaning |
| --- | --- |
| `INVALID_FORMAT`, `UNSUPPORTED_VERSION`, `INVALID_JSON` | not a plan of this format and version |
| `INVALID_TYPE`, `INVALID_VALUE`, `OUT_OF_RANGE`, `MISSING_KEY`, `UNKNOWN_KEY` | a value has the wrong shape (the pointer says which) |
| `INVALID_SYMBOL`, `DUPLICATE_SYMBOL` | a symbol is malformed, or defined twice |
| `UNKNOWN_REFERENCE`, `INVALID_REFERENCE`, `AMBIGUOUS_REFERENCE`, `REFERENCE_CYCLE` | references |
| `INVALID_EXPRESSION`, `UNKNOWN_VARIABLE`, `EXPRESSION_ERROR` | expressions (syntax, names, evaluation) |
| `PATTERN_UNSATISFIABLE` | a scatter cannot place its points |
| `UNKNOWN_OPERATION`, `SELECTION_MISMATCH`, `OPERATION_FAILED` | operations |
| `NAME_CONFLICT` | a name or id is taken |
| `REVISION_MISMATCH` | `expected_revision` is not the scene's revision |
| `IDEMPOTENCY_KEY_REUSED`, `IDEMPOTENCY_CONFLICT` | [idempotency](#retries-and-idempotency) |
| `LIMIT_EXCEEDED` | a [limit](#limits) |
| `UNSAFE_PATH` | a texture image outside the project folder |
| `TIMEOUT`, `CANCELLED` | stopped; rolled back |
| `NO_CAMERA` | a camera operation without a camera |

Warnings (`result.warnings`): `NOT_BOUND` (checked without a scene), `NAMES_CHANGED`,
`SCATTER_FEWER`, `EMPTY_SELECTION` (an update selected nothing).

## Retries and idempotency

- **`expected_revision`** makes a plan apply only to the scene its author read: after any
  other change, it fails with `REVISION_MISMATCH` and changes nothing.
- **`idempotency_key`** makes a resent plan apply once. The ids of the objects, materials
  and textures it makes are derived from the key and their place in the plan (UUID 5), so
  a plan already applied is recognized even by another process (the CLI, a restarted MCP
  server): it reports `status: "already_applied"` with the same ids. Reusing a key for a
  different plan in the same session is `IDEMPOTENCY_KEY_REUSED`; finding only some of the
  plan's objects (it was applied, then partly undone or deleted) is
  `IDEMPOTENCY_CONFLICT`. Use a new key to apply a plan again.

## Large plans: chunks

A plan is one transaction, so memory for its undo step grows with it. For very large
plans, `"options": {"chunk_size": 5000}` (with an `idempotency_key`) applies the objects in
chunks of that many, **one undo step per chunk** (labelled `"label (chunk k/n)"`), after a
history checkpoint named `plan:KEY:before`:

- if a chunk fails, the chunks before it stay applied (`on_chunk_failure: "keep"`) and the
  error's `details.chunks` says how far it got; **resending the same plan resumes**: chunks
  whose objects are all present are skipped. With `"rollback"`, the scene returns to the
  checkpoint instead;
- `history.goto("plan:KEY:before")` undoes a completed chunked plan as a whole.

Chunking is opt-in: it gives up the plan's atomicity, deliberately, for bounded steps.

## Limits

`PlanLimits` bounds what one plan may cost; a plan beyond a limit fails at the check with
`LIMIT_EXCEEDED`, before anything is built. Defaults:

| Limit | Default |
| --- | --- |
| `max_plan_bytes` (JSON text) | 16 MiB |
| `max_objects` (made by one plan) | 100,000 |
| `max_scene_objects` (in the scene afterwards) | 1,000,000 |
| `max_pattern_count` (instances of one pattern) | 100,000 |
| `max_depth` (nesting of children and prefabs) | 32 |
| `max_operations` | 10,000 |
| `max_queries` | 100 |
| `max_segments` (of a primitive) | 512 |
| `max_inline_vertices` (in `data` meshes) | 200,000 |
| `max_triangles` (of the new objects) | 50,000,000 |
| `max_new_mesh_bytes` | 1 GiB |
| `max_materials` (materials and textures) | 1,000 |

```python
from aether3d import PlanLimits

scene.execute_plan(plan, limits=PlanLimits(max_objects=20_000), timeout=60)
```

The CLI takes `--max-objects` and `--timeout`; the MCP server takes `--max-objects` and
`--timeout` for every client.

## The command line

```text
aether3d execute-plan PROJECT PLAN|-  [--dry-run | --validate] [--output OUT]
                                       [--history] [--timeout S] [--max-objects N]
                                       [--progress] [--json]
aether3d plan validate PLAN [--project PROJECT] [--json]
aether3d plan schema
```

`execute-plan` loads the project, applies the plan as one undo step and saves the project
-- atomically, to `--output` or in place. A project with a saved history (or `--history`)
gets the step in its history file, so `aether3d history undo` takes the plan back.
`--progress` reports on stderr. A plan with problems exits with status 1 and an error
report (`--json`: the codes and pointers of every problem); `plan validate` exits with
status 3 when the plan has problems.

## Python API

| Function | |
| --- | --- |
| `Scene.execute_plan(plan, **options)` | apply (or check, or dry-run) a plan; `PlanResult` |
| `aether3d.execute_plan(scene, plan, **options)` | the same |
| `aether3d.validate_plan(plan, scene=None)` | check without applying; `PlanResult` |
| `aether3d.plans.plan_schema()` | the JSON Schema (draft 2020-12) |
| `aether3d.plans.select_objects(scene, selector)` | the objects a selector names |
| `aether3d.plans.with_selections(scene, request)` | a spatial query request with selectors resolved |
| `PlanResult`, `PlanError`, `PlanLimits` | results, errors, limits |

## Performance

Measured with `benchmarks/bench_plans.py` on an Intel i9-12900H laptop (20 logical CPUs,
15 GiB), Arch Linux, Python 3.14.8, NumPy 2.5.4, NVIDIA RTX 4060 Laptop GPU (Vulkan),
on 2026-10-15. **0.10.0** is the same benchmark run against Aether3D 0.10.0 (the `main`
branch before this release), for the methods that existed then.

**Workload:** N blocks, 0.9 m cubes on a grid 100 wide, each a mesh object with its own
name and position; every method makes the same objects (the benchmark checks their names
and positions). Each method runs in a
fresh process: one warm-up build, then measured builds into new scenes, 5 at 100 and
1,000 blocks and 2 at 10,000; the tables give the median, with the range when it is
wider than a few percent. Peak RSS is the process's (Linux `VmHWM`), including Python and
NumPy (about 48 MiB).

### Building N blocks from Python

| Method | 100 | 1,000 | 10,000 | 10,000 in 0.10.0 | Undo steps kept (10,000) | Peak RSS (10,000) |
| --- | --- | --- | --- | --- | --- | --- |
| `add_cube` per block | 10.4 ms | 118 ms | 1,925 ms | 2,134 ms | 256 of 10,000 | 129 MiB |
| `add_object` per block (shared mesh) | 2.0 ms | 25.1 ms | 920 ms | 1,010 ms | 256 of 10,000 | 104 MiB |
| the same in one `transaction` | 1.8 ms | 17.4 ms | 379 ms | 452 ms | 1 | 66 MiB |
| one procedural script | 5.0 ms | 52.9 ms | 820 ms | 5,977 ms | 2 | 84 MiB |
| **one plan** | **1.4 ms** | **9.7 ms** | **105 ms** | -- | **1** | **70 MiB** |

At 10,000 blocks the plan builds about 96,000 blocks per second: 18x faster than
`add_cube`, 8.8x faster than `add_object`, 3.6x faster than the same calls in a
transaction. At 100,000 blocks (the default `max_objects`) it took 1.11 and 1.13 s in two
runs (about 89,000 per second; peak RSS 268 MiB; an undo step of 29.6 MB). Per-object
calls also make one undo step each, so past the history's limit (256 steps) the first
creations can no longer be undone, and their 256 steps take 40.6 MB; the plan is one
step of 3.0 MB.

### From the command line and over MCP

| Method | 100 | 1,000 | 10,000 |
| --- | --- | --- | --- |
| `aether3d procedural run` per block | 18.4 s (measured) | 274 s (projected) | 11,398 s = 3.2 h (projected) |
| one `aether3d procedural run` script | 0.19 s | 0.28 s | 2.64 s (0.10.0: 7.86 s) |
| one `aether3d execute-plan` | 0.18 s | 0.24 s | 0.77 s |
| one MCP `execute_scene_plan` call per block | 0.36 s | 2.15 s | 21.9 s |
| one MCP `execute_scene_plan` call | 4.1 ms (3.1-6.1) | 11.9 ms | 102 ms |

A CLI call starts Python, imports Aether3D, loads the project, changes it and saves it:
about 0.18 s for a small project here, growing with the project: 0.27 s per call on a
project of 1,000 blocks and 1.14 s on one of 10,000 (the projections multiply those
measured times by N, so they are lower bounds). An MCP tool call costs about 1.2 ms over
stdio (a trivial call), and a plan of 10,000 objects about 9 ms more as a tool call than
in the server; one call per object costs 1.7-1.9 ms each (3.7 ms at 100 objects, right
after the server started).

### Where the time goes (10,000 blocks, one plan)

| Stage | Time |
| --- | --- |
| checking the plan without a scene (`validate_plan`) | 0.08 ms |
| parse and check | 0.18 ms |
| bind: the grid's layout, names, references | 25.0 ms |
| objects: creating them in bulk, with their undo record | 62.2 ms |
| apply (resources, objects, operations) | 68.7 ms |
| total | 94 ms |
| geometry generation, for comparison: 10,000 `create_cube` calls (what `add_cube` pays) | 835 ms |
| undo / redo of the whole plan | 2.5 ms / 2.8 ms |
| first `overlaps` query (builds the spatial index) / again | 175 ms / 34.9 ms |
| `raycast` / `nearest` (5 objects) | 7.4 ms / 5.5 ms |
| GPU rasterizer, 1280x720, first render after the device exists / warm | 139 ms / 120 ms, **1 draw call** |
| CPU rasterizer, 640x360 | 0.73 s |

- **Creation** costs about 6 us per object (the undo record, ids, the appearance) and
  binding a grid about 2.5 us. An expression evaluated per instance costs about 3 us each:
  placing the same blocks with `"repeat"` and two expressions
  (`[{"expr": "i % 100"}, {"expr": "floor(i / 100)"}, 0.45]`) binds in 82 ms instead of
  25 ms. Let patterns do the placement when they can.
- **Geometry generation is not scene construction.** `add_cube` generates a new mesh per
  block (84 us each); plans share one mesh per definition, which also keeps memory low
  (70 MiB for 10,000 blocks against 129 MiB).
- **The garbage collector waits.** A plan pauses Python's cyclic garbage collector while
  it runs. In a process holding many objects (a long-running MCP server with a large
  scene, a test session), a full collection triggered by the plan's allocations walked
  all of them: an 8,000-object plan in a process holding a million objects took 256-278 ms,
  and takes 117-121 ms with the pause.
- **Rendering is a separate stage.** The GPU draws the 10,000 blocks in one instanced
  draw call; a warm frame takes 120 ms, of which the GPU's work is about 4 ms and the rest
  the per-frame scene extraction and upload on the CPU. Path tracing does not depend on
  how a scene was built and is not measured here (see `bench_pathtrace.py` and
  `bench_gpu_pathtrace.py`). The process that renders on the GPU peaks at 608 MiB, most of
  it the GPU driver.
- **Edits stay small.** Changing one object of a large scene with a plan costs about
  0.12 ms at any size (it was 3.8 ms at 20,000 objects before 0.11;
  `tests/test_plans_performance.py` keeps it so).

| Example plan | Objects | Build (median of 2) | Of which queries | Undo step | GPU 1280x720 (warm) |
| --- | --- | --- | --- | --- | --- |
| castle | 2,190 | 105 ms | 70 ms | 1.3 MB | 57 ms, 10 draw calls |
| village | 11,270 | 351 ms | 166 ms | 6.9 MB | 131 ms, 20 draw calls |

What 0.11 fixed on the way, found by profiling: automatic names were found by scanning
the taken names (N creations were quadratic); edit plans walked the whole scene twice;
procedural scripts scanned every object at every step (5,977 ms for 10,000 blocks, now
820 ms); renders built a material record per object; and plans let full garbage
collections run over the whole process.

To reproduce: `python benchmarks/bench_plans.py --json results.json` (about five
minutes; `--quick` for 100 and 1,000 blocks), and for a baseline,
`PYTHONPATH=old-aether3d/src python benchmarks/bench_plans.py --no-mcp`.
