# Aether3D sculpting: a guide for agents

Sculpt an object's mesh with brush strokes, masks, symmetry, dynamic topology and
remeshing, all as **data**: one `aether3d.sculpt-script` document, applied as one undo
step (everything or nothing). Nothing in a script is executed as code.

## A script

```json
{"format": "aether3d.sculpt-script", "version": 1, "label": "Shape the head",
 "limits": {"max_dabs": 20000, "max_triangles": 1000000, "max_seconds": 60},
 "sculpts": [
  {"object": "Head", "require": "closed",
   "operations": [
    {"op": "set_mask", "value": 1, "box": [[-2, -2, -2], [2, 2, -0.2]]},
    {"op": "stroke",
     "brush": {"kind": "draw", "radius": 0.15, "strength": 0.3, "symmetry": "x"},
     "path": [[0.3, -0.9, 0.2], [0.35, -0.85, 0.35]]},
    {"op": "stroke",
     "brush": {"kind": "clay", "radius": 0.2, "strength": 0.4,
               "dyntopo": {"detail": 0.25}},
     "path": [{"position": [0, -1, 0.5], "pressure": 0.4},
              {"position": [0, -0.9, 0.7], "pressure": 1.0}]},
    {"op": "clear_mask"},
    {"op": "check"}
   ]}
 ]}
```

Run it with `run_script` (MCP), `aether3d sculpt run PROJECT script.json` (CLI) or
`aether3d.run_sculpt_script(scene, script)` (Python). The full schema is the resource
`aether3d://schema/sculpt-script`; `capabilities {"topic": "sculpt"}` lists brushes,
operations and tiers.

## Strokes

* `path`: points in **world** space (`"space": "local"` for the object's own
  coordinates), each `[x, y, z]` or `{"position": [x, y, z], "pressure": p}` (0..1).
  Dabs are placed every `spacing * radius` along the path (`"resample": false` keeps
  only the given points).
* Every dab is moved onto the surface first (`"project": "nearest"`, the default), so
  approximate points work. `"project": "ray"` casts along `"view"` (a direction) and
  `"none"` uses the points as they are.
* Brush `radius` is in **world units**; `strength` 0..1 (most brushes: displacement
  `strength * radius` at the centre); `direction: -1` inverts (draw lowers, clay digs).
* Brushes: `draw`, `inflate`, `clay`, `flatten`, `fill`, `scrape`, `pinch`, `crease`,
  `layer`, `grab`, `elastic`, `snake_hook`, `twist`, `smooth` (`smooth_method`:
  `laplacian`, `taubin`, `relax`), `mask` (paints the mask). `grab`, `elastic` and
  `snake_hook` snap only their first dab to the surface; the rest of the path is the
  motion, so a path leaving the surface pulls the geometry out with it.
* `falloff`: `smooth`, `smoother`, `sphere`, `root`, `sharp`, `linear`, `constant`, or a
  curve of `[x, y]` points; `hardness` keeps full strength up to a fraction of the radius.
* `symmetry`: `"x"`, `"xy"`, ...: every dab is mirrored across the object's local planes
  through its origin. Vertices on a plane stay on it; a symmetric mesh stays symmetric.
* `pressure` (brush): which of `strength`, `radius`, `both` or `none` the path's
  pressure scales.
* `automask`: `topology` (only the surface connected to the dab centre), `face_groups`
  (only the face group under it), `boundary` (open edges held).

## Masks

`set_mask` (`value`, `mode`: `set`, `add`, `subtract`) on `vertices` (ids), a `sphere`
`[[x, y, z], r]`, a `box` `[[x0, y0, z0], [x1, y1, z1]]` or face `groups`, in the
object's **local** coordinates; `invert_mask`, `clear_mask`. Masked vertices (1) do not
move under brushes or relaxation, and dynamic topology does not change them (symmetrize
in `mirror` mode rebuilds the other side whatever its mask).

## Topology

* **Dynamic topology**: before each dab, a brush's `"dyntopo": {"detail": d}` splits
  the edges inside the brush longer than about `d * radius` (`"mode": "constant"`: `d`
  in world units), then collapses shorter ones, flips and relaxes in and around it.
  UVs are interpolated exactly on splits; seams, hard edges, face-group borders and
  open boundaries are kept (split along, never collapsed across). Not for `grab`,
  `elastic`, `layer` or `mask`.
* `{"op": "remesh", "target_length": L}`: isotropic remeshing of the whole mesh toward
  edges of length `L` (world units), keeping the same features; reports edge-length
  statistics.
* `{"op": "voxel_remesh", "voxel_size": h}`: rebuilds the mesh from a voxel grid. The
  mesh must be closed (`VOXEL_NOT_WATERTIGHT` otherwise: close holes first, or use
  `"mode": "shell", "thickness": t`). Overlapping parts fuse into one surface;
  inward-facing parts become cavities. **Destructive**: new vertices, UVs dropped (a
  `VOXEL_UVS_DROPPED` warning), groups and mask carried over from the nearest point,
  details under about two voxels lost. It reports validity metrics before and after.
* `{"op": "symmetrize", "axis": "x", "direction": "+"}`: copies the `+x` side onto the
  `-x` side -- exactly, vertex by vertex, when the sides' vertices pair up; otherwise it
  cuts the mesh at the plane and mirrors the `+x` half (`"mode": "positions"` or
  `"mirror"` forces one).
* Not offered: quad remeshing and multiresolution sculpting.

## Validity

Tiers, weakest first: `valid` (finite positions) < `manifold` (every edge has one or
two triangles, every vertex one fan, consistent winding, no degenerate triangle) <
`closed` (manifold, no boundary edges) < `solid` (closed, positive volume, no
self-intersection; needs the exact self-intersection test). Give each sculpt a
`"require"` tier: a result below it is refused (`BELOW_VALIDITY_TIER`) and nothing
changes. `{"op": "check"}` reports the current state; the commit's `report` the result.
`inspect_mesh` (MCP) and `aether3d sculpt check` measure any object.

## Errors and limits

Errors are structured: `code`, `message`, `location` (`sculpts[0].operations[2]`) and
a `hint`; nothing is changed. Codes: `INVALID_SCRIPT`, `INVALID_ARGUMENT`,
`ValidationError` (a brush or stroke setting), `OBJECT_NOT_FOUND`, `NO_MESH`,
`PROCEDURAL_OBJECT` (bake it or sculpt its source), `BUDGET_EXCEEDED` (a `limits`
value), `BELOW_VALIDITY_TIER`, `REMESH_TOO_DENSE`, `TOO_MANY_ELEMENTS`,
`VOXEL_NOT_WATERTIGHT`, `VOXEL_INSIDE_OUT`, `VOXEL_EMPTY`, `VOXEL_GRID_TOO_LARGE`,
`SYMMETRIZE_NO_SOURCE`.

## Look at the result

After sculpting, `observe_scene` (a close-up: `{"view": {"focus": {"objects": "Head"}},
"focus": "Head"}`) and `diagnose_scene` with a `mesh_validity` check (`{"check":
"mesh_validity", "objects": "Head", "tier": "closed"}`) confirm what the mesh looks
like and that it stayed sound. Undo the whole script with `history` (`undo`) if it is
not what you wanted.
