# Aether3D scene plans: a guide for agents

A **scene plan** builds or changes a scene in one call: checked as a whole, then applied
as one undo step, all or nothing. Use one plan for a whole structure instead of many small
calls; thousands of objects cost one call.

```json
{
  "format": "aether3d.scene-plan", "version": 1, "label": "Courtyard",
  "materials": {"stone": {"preset": "stone"}, "red": {"base_color": "#a03020"}},
  "meshes": {"block": {"primitive": "cube", "size": 1}},
  "objects": {
    "ground": {"mesh": {"primitive": "plane", "size": 40}, "color": "#4f7f3a"},
    "wall": {"mesh": "@block", "material": "@stone", "position": [-5, 0, 0.5],
             "grid": {"count": [11, 1, 3], "spacing": 1}},
    "pillars": {"mesh": {"primitive": "cylinder", "radius": 0.3, "depth": 3},
                "circle": {"count": 8, "radius": 6, "z": 1.5}, "material": "@red"},
    "sun": {"light": {"type": "directional", "intensity": 2}, "direction": [-0.4, 0.5, -1]},
    "cam": {"camera": {"fov": 45}, "position": [14, -16, 9], "look_at": [0, 0, 1]}
  },
  "queries": [{"query": "overlaps", "objects": "@pillars"}]
}
```

## Conventions

- +Z is up, units are meters, objects face -Y (cameras look along -Y).
- `rotation_euler` is in degrees (XYZ); `scale` is a number or `[x, y, z]`.
- Colors are sRGB: `"#rrggbb"` or `[r, g, b]` in 0..1.
- A cube of size 1 spans -0.5..0.5: stand it on the ground at `z = 0.5`.

## Symbols and references

- Every key in `meshes`, `textures`, `materials`, `geometries`, `prefabs` and `objects` is a
  **symbol**. Symbols are unique across sections: ASCII letters, digits, `_ - .`.
- Reference what the plan makes with `"@symbol"`:
  - `"@towers"` is every instance;
  - `"@towers[2]"` is one instance (negative indexes count from the end);
  - `"@towers[2]/roof"` is a child, and `"@towers/roof"` is that child of every instance.
- Existing objects are referenced by id or name, without `@` (`"Ground"`), or as
  `{"id": ...}` / `{"name": ...}`.

## Objects

- **Kind:** at most one of `mesh` (`"@symbol"` or an inline `{"primitive": ...}`),
  `geometry` (a procedural node), `prefab`, `camera`, `light`. With none of them, the
  object is an empty group.
- **Placement and look:** `name`, `parent` (top-level objects), `position`,
  `rotation_euler`, `scale`, `look_at` (a point or an object), `material` (or `color`),
  `visible`, `tags`, `metadata`.
- **children:** objects created under every instance, by symbol.
- **Mesh primitives:**
  - `cube` (`size`);
  - `box` (`size [x, y, z]`);
  - `sphere` (`radius`, `segments`, `rings`);
  - `plane` (`size`);
  - `cylinder` (`radius`, `depth`, `segments`);
  - `cone` (`radius_bottom`, `radius_top`, `depth`, `segments`);
  - `torus`, `grid`, `uv_sphere`.

  Identical definitions share one mesh, and the GPU then draws all those objects in one
  instanced draw call.
- **Names:**
  - templates: `{i}`, `{i:3}` (zero-padded), `{n}`, `{x}`, `{y}`, `{z}`, `{parent}`;
  - defaults: the symbol, `"symbol i"` for patterns, `"Parent/child"` for children;
  - names must be free: a taken name is an error, unless `"options": {"on_name_conflict": "rename"}`.

## Patterns (one per object)

Each pattern turns one definition into many instances:

| Pattern | Makes |
| --- | --- |
| `"grid": {"count": [nx, ny, nz], "spacing": s, "center": true}` | a box of instances (x fastest) |
| `"array": {"count": n, "offset": [dx, dy, dz], "rotation_step": [..], "scale_step": f}` | a row |
| `"circle": {"count": n, "radius": r, "orient": "outward"}` | a ring around +Z (`inward`, `tangent`) |
| `"line": {"from": [..], "to": [..], "count": n}` | evenly spaced, ends included |
| `"scatter": {"count": n, "box": {"min": [x, y], "max": [x, y]}, "seed": 1, "min_distance": d}` | random but reproducible |
| `"instances": [{"position": [..], "rotation_euler": [..], "name": ".."}, ...]` | explicit instances |
| `"repeat": n` | `n` instances, placed with expressions over `i` |

- `position` adds to the pattern's positions.
- **where** skips instances: `"where": {"expr": "not (x >= 2 and x <= 3 and z <= 1)"}`
  leaves a doorway in a wall of blocks. Names keep the pattern's indexes; `"@wall[k]"`
  counts the instances made.
- **vary** changes fields per instance (`color`, `material`, `mesh`, `scale`,
  `rotation_euler`, `offset`, `visible`, `tags`):
  `{"cycle": [...]}`, `{"choose": [...], "seed": 1}`, `{"uniform": [low, high], "seed": 2}`,
  `{"linear": [start, end]}`, `{"values": [one per instance]}` or `{"expr": "..."}`.
- **Expressions** (`{"expr": "i * 2"}`) use plan `parameters` and, per instance, `i`,
  `n`, `u` (0..1), `x`, `y`, `z` (grid indexes) and `angle` (circle); math functions
  take degrees.

## Prefabs: reusable structures

```json
"prefabs": {"house": {"children": {
  "walls": {"mesh": "@box", "position": [0, 0, 1.5], "material": "@plaster"},
  "roof": {"mesh": "@roof", "position": [0, 0, 3.5], "material": "@tiles"}}}},
"objects": {"street": {"prefab": "@house", "array": {"count": 12, "offset": [8, 0, 0]}}}
```

A prefab is a template, not an object: reach its copies through the objects that use it
(`"@street/roof"`, `"@street[3]/walls"`). Prefabs can use other prefabs. The example plans
(`aether3d://examples/castle`, `aether3d://examples/village`) show larger structures.

## Operations (after building, in order)

- `{"op": "update", "select": ..., "set": {...}}`. `set` takes: `position`, `translate`,
  `rotation_euler`, `scale`, `scale_by`, `color`, `material`, `visible`, `tags`,
  `add_tags`, `remove_tags`, `metadata` (`null` removes a key), `parent`
  (+ `keep_world_transform`), `name` (one object), `look_at`.
- `{"op": "delete", "select": ..., "recursive": true}`. Selecting by filters needs
  `"expect": N`.
- `{"op": "set_parameters", "values": {...}}`, `{"op": "frame_camera", "targets": ...}`,
  `{"op": "set_active_camera", "camera": "@cam"}`.
- `{"op": "cad", "script": {"format": "aether3d.cad-script", ...}}`: precision models
  (parametric sketches and parts, buildings, drawings) in the plan's step; a CAD item
  shown with `{"op": "show", "item": ...}` becomes an object later operations can select
  by its name (guide: `aether3d://cad/guide`).
- **Selectors:** a reference, a list of them, or
  `{"refs", "ids", "names", "tags", "any_tags", "name_glob", "kind", "under", "material",
  "expect"}`.

## Checking your work

- `queries` in a plan are answered right after it, for example
  `{"query": "overlaps", "objects": "@houses/walls"}` or
  `{"query": "raycast", "origin": [0, 0, 50], "direction": [0, 0, -1]}`.
- Object lists in queries (`objects`, `against`, `exclude`) also take a selector:
  `{"query": "overlaps", "objects": {"name_glob": "Tree */*"}, "against": {"under": "keep"}}`.
- `spatial_query` answers the same queries any time: `describe`, `bounds`, `find`,
  `raycast`, `closest_point`, `nearest`, `contains`, `distance`, `overlaps`, `region`,
  `pick`, `in_view`, `line_of_sight`, `visible_from`, `relation`.
- To fix a mistake, send a small plan of operations; to step back, use `history` undo.
- `render_preview` shows the scene: the GPU rasterizer when it works, the CPU one otherwise.

## Errors and safety

- **Errors:** a failing plan changes nothing. Every problem comes back with a `code`, a
  JSON `pointer` (`/objects/wall/grid/count`), the `operation` index and a `hint`. Fix
  them all and resend.
- **Retries and concurrency:**
  - `"idempotency_key"` makes a resent plan apply once;
  - `"expected_revision"` refuses to apply to a scene that changed since you read it.
- **Large plans:** `"options": {"chunk_size": 5000}` (with an idempotency key) commits in
  steps that a re-run resumes.
