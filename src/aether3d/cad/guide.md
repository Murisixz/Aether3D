# Aether3D precision modeling: a guide for agents

Build exact, parametric models -- sketches with constraints and dimensions, solid parts
made of features, assemblies placed by mates, buildings with walls, openings, rooms,
slabs, roofs, stairs, columns and beams, and drawings of all of them -- as **data**: one
`aether3d.cad-script` document, applied as one undo step (everything or nothing).
Nothing in a script is executed as code (`"=width / 2"` is arithmetic over parameters,
evaluated by a restricted evaluator) and a script never reads or writes a file.

Run it with `run_script` (MCP), `aether3d cad run PROJECT script.json` (CLI) or
`aether3d.cad.script.run_cad_script(scene, script)` (Python). The full schema is the
resource `aether3d://schema/cad-script`; `capabilities {"topic": "cad"}` lists item
kinds, feature, constraint and mate types, views and formats.

## Units and tolerances

* Every length in a script is a number in the document's unit (`{"op": "units",
  "length": "mm"}` first) or a quantity: `"80 mm"`, `"3/8 in"`, `"2' 6\""`, `"30 deg"`.
  Sketch point positions and radii included.
* `"=expr"` evaluates over parameters (`{"op": "parameter", "name": "width", "value":
  "80 mm"}`): `+ - * /`, powers, parentheses and a few functions (`sqrt`, `hypot`,
  `sin`, `cos`, `tan`, `min`, `max`, `abs`, ...; `capabilities` lists them) -- nothing
  else. Changing a parameter regenerates exactly what depends on it.
* Geometry is exact (planes, cylinders, cones, spheres, tori, NURBS) and kept in metres;
  tolerances scale with the model (`info` reports them). Display meshes are
  tessellations within a chord tolerance -- never confuse the two: measure with
  `measure`, not with the mesh.

## A part

```json
{"format": "aether3d.cad-script", "version": 1, "label": "Bracket",
 "operations": [
  {"op": "units", "length": "mm"},
  {"op": "parameter", "name": "width", "value": "80 mm"},
  {"op": "add", "kind": "sketch", "name": "Profile", "definition": {"plane": "XY",
   "entities": [
     {"id": "p1", "type": "point", "at": [0, 0]}, {"id": "p2", "type": "point", "at": [80, 0]},
     {"id": "p3", "type": "point", "at": [80, 40]}, {"id": "p4", "type": "point", "at": [0, 40]},
     {"id": "l1", "type": "line", "start": "p1", "end": "p2"},
     {"id": "l2", "type": "line", "start": "p2", "end": "p3"},
     {"id": "l3", "type": "line", "start": "p3", "end": "p4"},
     {"id": "l4", "type": "line", "start": "p4", "end": "p1"}],
   "constraints": [
     {"id": "f", "type": "fix", "refs": ["p1"]},
     {"id": "h1", "type": "horizontal", "refs": ["l1"]},
     {"id": "v1", "type": "vertical", "refs": ["l2"]},
     {"id": "h2", "type": "horizontal", "refs": ["l3"]},
     {"id": "v2", "type": "vertical", "refs": ["l4"]},
     {"id": "w", "type": "length", "refs": ["l1"], "value": "=width"},
     {"id": "d", "type": "length", "refs": ["l2"], "value": 40}]}},
  {"op": "add", "kind": "part", "name": "Bracket", "definition": {"features": [
     {"type": "extrude", "id": "base", "sketch": "Profile",
      "extent": {"type": "blind", "distance": 10}},
     {"type": "hole", "id": "h", "face": "base:end", "points": [[20, 20, 10], [60, 20, 10]],
      "diameter": 8, "kind": "counterbore", "cbore_diameter": 14, "cbore_depth": 4},
     {"type": "fillet", "id": "round", "edges": [{"edge": ["base:end", "base:side/l2"]}],
      "radius": 3}]}},
  {"op": "show", "item": "Bracket"},
  {"op": "expect", "info": "Profile", "checks": {"status": "fully_constrained"}},
  {"op": "expect", "info": "Bracket", "checks": {"bodies.main.valid": true}}
 ]}
```

* **Sketches** solve on every change. `info` on a sketch reports `status`
  (`fully_constrained`; `under_constrained` with the `free` points and radii;
  `redundant` -- consistent but repeated constraints, listed in `redundant`), `dof` and
  residuals. A sketch whose constraints contradict each other is refused
  (`SKETCH_CONFLICT`, with the conflicting constraints) and nothing changes. Dimension values: `distance`, `length`, `angle`, `radius`, `diameter`;
  `"driving": false` makes a reference (measured) dimension.
* **Features** (`type`): `extrude` (`extent`: blind, symmetric, two_sided, through_all;
  `draft`; `operation`: new, add, cut, intersect), `revolve`, `sweep`, `loft`, `hole`
  (simple, counterbore, countersink, drill point), `fillet`, `chamfer` (straight edges
  between planes), `shell`, `pattern` (linear, circular), `mirror`, `transform`,
  `boolean`, primitives (`box`, `cylinder`, `sphere`, `cone`, `torus`), `solid` (an
  imported B-rep). Unsupported configurations are refused with the reason, never
  approximated.
* **Names** are persistent: faces are `"<feature>:<role>"` (`base:end`, `base:side/l2`
  -- the side swept by sketch line `l2` --, `h.0:rev/wall.0` -- a wall of the hole at
  feature `h`'s first point); a face split by a later feature gets `#0`, `#1`, ...;
  edges are the pair of faces they separate. When a name
  matches several pieces, add `"near": [x, y, z]`. Errors `REFERENCE_LOST` and
  `REFERENCE_AMBIGUOUS` list the candidates. `inspect_cad` on a part lists its faces
  by name.

## Editing

`append`, `update` and `delete` edit one list of a definition by element `id` (sketch
`entities` and `constraints`, part `features`, floor `walls`, `openings` and `rooms`,
assembly `components` and `mates`, building `levels` by `name`); `merge` replaces
top-level keys; `replace` the whole definition. Example -- change a dimension and move a
hole:

```json
{"op": "update", "item": "Profile", "list": "constraints", "id": "w", "changes": {"value": 100}},
{"op": "update", "item": "Bracket", "list": "features", "id": "h",
 "changes": {"points": [[25, 20, 10], [75, 20, 10]]}}
```

Only what depends on the change regenerates (`regeneration` in the result counts it).

## Buildings

```json
{"op": "units", "length": "m"},
{"op": "add", "kind": "building", "name": "House", "definition": {
  "levels": [{"name": "Ground", "elevation": 0, "height": 3}, {"name": "First", "elevation": 3, "height": 3}],
  "wall_types": {"Ext": {"thickness": "300 mm", "function": "exterior"}, "Int": {"thickness": "120 mm"}},
  "families": {"D90": {"kind": "door", "width": "900 mm", "height": "2100 mm"},
               "W12": {"kind": "window", "width": "1200 mm", "height": "1400 mm", "sill": "900 mm"}},
  "criteria": {"riser_max": "190 mm", "tread_min": "250 mm"}}},
{"op": "add", "kind": "floor", "name": "Ground floor", "definition": {
  "building": "House", "level": "Ground",
  "walls": [{"id": "s", "start": [0, 0], "end": [10, 0], "type": "Ext"},
            {"id": "e", "start": [10, 0], "end": [10, 8], "type": "Ext"},
            {"id": "n", "start": [10, 8], "end": [0, 8], "type": "Ext"},
            {"id": "w", "start": [0, 8], "end": [0, 0], "type": "Ext"},
            {"id": "mid", "start": [6, 0], "end": [6, 8], "type": "Int"}],
  "openings": [{"id": "front", "wall": "s", "position": 3, "family": "D90"},
               {"id": "east", "wall": "e", "position": 4, "family": "W12"}],
  "rooms": [{"id": "living", "name": "Living", "number": "001", "point": [3, 4]}]}},
{"op": "add", "kind": "slab", "name": "Ground slab", "definition": {
  "building": "House", "level": "Ground", "thickness": 0.25, "outline": {"floor": "Ground floor"}}},
{"op": "add", "kind": "roof", "name": "Roof", "definition": {
  "building": "House", "level": "First", "type": "gable", "slope": "30 deg",
  "overhang": 0.5, "ridge": "x", "thickness": 0.2, "outline": {"floor": "Ground floor"}}},
{"op": "schedule", "building": "House", "kind": "walls"}
```

Walls join exactly in plan (corners, T junctions, crossings); openings must fit their
wall (`OPENING_OUTSIDE_WALL`, `OPENING_TOO_TALL`, `OPENING_OVERLAP`);
rooms are the regions bounded by wall faces around their points, with net areas.
Quantities follow the IFC base quantity definitions. Stairs are checked against the
`criteria` *you* give -- Aether3D assumes no building code and claims no compliance.

## Drawings

```json
{"op": "add", "kind": "drawing", "name": "Ground plan", "definition": {
  "view": "plan", "items": ["Ground floor", "Ground slab"], "level": "Ground",
  "cut_height": "1.2 m", "scale": "1:100"}}
```

Views: `plan`, `section` (`plane`: `{origin, normal}`, the viewer on the normal's side),
`top`, `front`, `back`, `left`, `right`, `bottom`, `iso`. Edges are the exact curves,
hidden lines come from ray casting, sections are exact cuts with hatching. Export with
`export_scene` (`.svg`, `.dxf`, `.png` and the drawing as `item`); `inspect_cad` with `image: "drawing"` returns one as an
image.

## Checking your work

* `measure` (read-only): one entity (a face's surface, area, radius, axis; an edge's
  length) or two (angle, parallel distance, axis distance, minimum distance with its
  bound). References: `{"part": "Bracket", "face": "base:end"}`, `{"part": ...,
  "edge": [faceA, faceB]}`, `{"sketch": "Profile", "entity": "l1"}`, `{"point": [x, y, z]}`.
* `expect` checks measured values or `info` values (dotted paths) and aborts the whole
  script -- rolling everything back -- when one fails. Put the checks that matter at the
  end of the script that builds the model.
* `inspect_cad` (MCP) reports an item: sketch status, DOF and conflicts; part validity,
  volume, mass properties, faces; building quantities and diagnostics. With an image it
  renders a drawing of the item; whether you can see that image depends on your client
  and model -- the numbers in the report are the authority.

## Exchange

`import_cad` and `export_scene` with a CAD suffix (MCP) or `aether3d cad import|export`
(CLI): STEP (exact
solids, AP214), IFC4 (buildings: walls with layer sets, openings, doors, windows,
spaces, slabs, roofs, stairs, columns, beams, with quantities and property sets), DXF
(2D: sketches in and out, drawings out), SVG and PNG (drawings). Imported files are
untrusted: strict parsers with size and entity limits; anything unsupported is refused
or skipped and reported, never guessed.

## Limits and errors

`limits`: `max_seconds` (regeneration included), `max_items`. Errors have a stable
`code`, a `location` (`operations[3]: part[Bracket].features[2](h)`), a `hint` and
`details`. Nothing is applied when a script fails.
