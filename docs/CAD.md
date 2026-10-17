# Precision modeling and CAD (0.15.0)

Exact, parametric models beside Aether3D's meshes: units and tolerances, sketches solved
under constraints and dimensions, solid parts built from features on an exact boundary
representation (B-rep), assemblies placed by mates, measurements, drawings, and STEP,
DXF and IFC exchange. Buildings -- levels, walls, openings, rooms, slabs, roofs, stairs,
columns, beams and their quantities -- are in [BUILDINGS.md](BUILDINGS.md).

Everything is reachable from Python (`scene.cad`), from JSON CAD scripts, the command line
(`aether3d cad ...`), scene plans (`{"op": "cad", ...}`) and MCP (`run_script`,
`inspect_cad`, `import_cad`, `export_scene`). The design record, with the research behind
it and the per-subsystem language decisions, is
[research/2026-10-stage15/DESIGN.md](../research/2026-10-stage15/DESIGN.md).

## Quick start

```python
from aether3d import Scene

scene = Scene()
doc = scene.cad
doc.set_units(length="mm")
doc.parameter("width", "80 mm")

sketch = doc.sketch("Profile")  # on the XY plane
lines = sketch.rectangle((0, 0), (80, 40))  # four lines, horizontal and vertical
sketch.constrain("fix", "p1")
sketch.dimension("length", lines[0], value="=width")
sketch.dimension("length", lines[1], value="40 mm")
print(doc.info("Profile")["status"])  # fully_constrained

part = doc.part("Plate")
part.extrude(sketch, 10, id="base")  # 10 mm, the document's unit
part.hole(
    "base:end",
    [[20, 20, 10], [60, 20, 10]],
    8,
    kind="counterbore",
    cbore_diameter=14,
    cbore_depth=4,
    id="holes",
)
part.fillet([{"edge": ["base:end", "base:side/r2"]}], 3, id="round")
m = part.measure("main")
print(m["valid"], round(m["volume"], 3))  # True 30088.053

doc.parameter("width", "100 mm")  # everything downstream regenerates
print(round(part.measure("main")["volume"], 3))  # 38088.053
gap = doc.measure(
    {"part": "Plate", "face": "holes.0:rev/wall.0"}, {"part": "Plate", "face": "holes.1:rev/wall.0"}
)
print(round(gap["axis_distance"], 9))  # 40.0

doc.drawing("Top", "top", ["Plate"])
svg = doc.result("Top").to_svg()  # also .to_dxf(), .to_png(dpi)
step = doc.export_step(["Plate"])  # exact B-rep, AP214
print(step.splitlines()[0], len(doc.import_step(step, name="Copy")))  # ISO-10303-21; 1
```

Every call that changes the document is one undo step (`scene.history.undo()`);
`with doc.batch("label"):` groups several into one, regenerated once at the end. A change
that cannot be regenerated -- a conflicting dimension, a hole off its face -- is refused
with a structured `CadError` and the document is left as it was.

## The document

`scene.cad` is the scene's CAD document: named **items** whose definitions are plain JSON
(no code), regenerated into results when they or what they read change.

| kind | definition | result |
| --- | --- | --- |
| `parameter` | `value` (quantity or `"=expr"`), `dimension` | a quantity |
| `frame` | `origin`, `z`, `x` (or relative to another frame) | a coordinate frame |
| `sketch` | `plane`, `entities`, `constraints` | the solved sketch, its regions |
| `part` | `features` (ordered), `materials`, `color` | named exact bodies |
| `assembly` | `components` (parts or assemblies), `mates` | placements, DOF, conflicts |
| `building`, `floor`, `slab`, `roof`, `stair`, `columns`, `beams` | [BUILDINGS.md](BUILDINGS.md) | elements and quantities |
| `drawing` | `view`, `items`, options | 2D linework, fills, text, dimensions |

* **Regeneration is incremental.** Each result is keyed by its definition and the keys of
  what it reads; only changed items and their dependents re-evaluate (`doc.stats` counts
  `evaluations` and `cache_hits`). Inside a part, a feature reruns only if it or an earlier
  feature changed.
* **Showing** an item (`doc.show(item)`) makes a scene object that follows it: its mesh is
  the item's tessellation at the display tolerance, regenerated with the item. Such an
  object's mesh cannot be edited directly (`CAD_DRIVEN`); `doc.convert_to_mesh(obj)`
  detaches it, keeping the current mesh and losing the exact geometry, the parametric
  updates and the face names -- it says so.
* **Files.** Projects store the CAD document (since format 8; older projects are migrated on
  load, see [FILE_FORMAT.md](FILE_FORMAT.md)): definitions only -- results are regenerated
  on load. The history file records CAD changes like any other.
* **Limits.** At most 100,000 items per document, 2,000 features per part; definitions are
  limited in depth, size and string length; every regeneration can carry a time budget
  and a cancellation callback (`doc.batch(..., deadline=..., should_stop=...)`).

## Units, quantities and tolerances

* A document has a **length unit** (`um`, `mm`, `cm`, `m`, `km`, `in`, `ft`, `yd`) and an
  **angle unit** (`deg`, `rad`, `arcmin`, `arcsec`, `gon`); numbers in definitions are in
  them. Quantities may be text: `"80 mm"`, `"3/8 in"`, `"1 1/2 in"`, `"2' 6\""` (feet
  and inches), `"12 m2"`, `"30 deg"`, `"45°30'15\""`, `"45 deg 30 arcmin"`. Conversions use exact rational factors; mixing dimensions (a length
  where an angle is due) is `UNIT_MISMATCH`.
* **Expressions** `"=width / 2 + 3 mm"` use document parameters, `+ - * / ** %`,
  comparisons and a few functions (`abs`, `min`, `max`, `clamp`, `round`, `floor`,
  `ceil`, `sqrt`, `hypot`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `degrees`,
  `radians`): they are parsed into a whitelisted syntax tree and
  evaluated with units -- never executed as Python.
* **Tolerances** are not scattered constants: `Tolerance.for_scale(size)` derives the
  resolution (distinct points), the solver tolerance and the angular tolerance from the
  model's size and float64's unit roundoff; the sketch report shows them. Display and
  analysis tessellations have their own chord and angle (`TessellationTolerance`;
  `doc.set_display_tolerance(...)`).
* Internally every length is in metres; formatting (`format_quantity`) gives decimal,
  architectural (`3'-4 1/2"`) or fractional text.

## Sketches

A sketch lives on a plane (`"XY"`, `"XZ"`, `"YZ"`, a frame item, or `{origin, z, x}`).
**Entities**: `point`, `line`, `circle`, `arc` (counter-clockwise), `spline` (B-spline
through control points); any may be `"construction": true`. **Constraints**:

| type | refs | type | refs |
| --- | --- | --- | --- |
| `coincident` | point, point / point, curve | `equal` | lines / circles |
| `horizontal`, `vertical` | line / two points | `midpoint` | point, line |
| `parallel`, `perpendicular`, `collinear` | two lines | `point_on` | point, curve |
| `tangent` | line or circle, circle or arc | `symmetric` | point, point, line or point |
| `concentric` | two circles or arcs | `fix` | an entity |
| `distance` (`along` x/y), `length`, `angle`, `radius`, `diameter` | | dimensions with a `value` | |

Dimensions are **driving** (they change the geometry) or **reference** (`"driving":
false`: measured, reported in `info["reference"]`). `SketchHandle` builds them
(`sketch.line`, `rectangle`, `polyline`, `circle`, `arc`, `spline`, `constrain`,
`dimension`, `set_value`, `remove`).

**The solver** compiles constraints to equations with analytic Jacobians, splits the
system into connected components and their Dulmage-Mendelsohn decomposition
(over-determined part, well-determined blocks in solving order, under-determined part),
and solves block by block with Gauss-Newton (minimum-norm steps keep under-constrained
geometry where it was) and Levenberg-Marquardt. It then analyses the result exactly:

* `status`: `fully_constrained`, `under_constrained` (with the `free` points and radii),
  `redundant` (consistent repeated constraints, listed with what they repeat), or the
  sketch is refused with `SKETCH_CONFLICT` naming each conflicting constraint, what it
  conflicts with, and the residual -- `{"constraint": "dd", "conflicts_with": ["d", "h1",
  "v1", "w"], "residual": 0.0042}`.
* `dof` (from the numerical rank of the Jacobian: SVD, or for large well-constrained
  components an exact check of the decomposition's diagonal blocks), the condition
  number, iterations, residuals.
* **Stable branches.** Where a constraint has two solutions (which side of a line, inner
  or outer tangency, the sign of an angle), the choice is recorded in the constraint
  when it is added and kept: a solve never jumps to the mirror image.
* **Incremental.** After a dimension edit only the pieces downstream of it are solved;
  the rest keeps its stored solution (a 5,000-constraint sketch re-solves one dimension
  in a few Newton iterations; [PERFORMANCE.md](PERFORMANCE.md#precision-modeling-015)).
* **Atomic.** A sketch that does not converge or conflicts changes nothing.

`doc.snap(sketch, x, y, radius=..., kinds=..., grid=...)` returns the best snap point for
a cursor: end points, then intersections, centres, midpoints, quadrants, the nearest
point on a curve, the grid -- exact.

## Parts and features

A part is an ordered list of features replayed on named bodies (`"main"` by default):

| feature | what it does |
| --- | --- |
| `extrude` | a sketch's regions along its normal: blind, symmetric, two-sided or through all; draft angle; `operation` new, add, cut, intersect |
| `revolve` | about a sketch line or an axis, any angle |
| `sweep` | a profile along a path of lines and tangent arcs (exact: segments sewn) |
| `loft` | ruled between two or more sections with matching loops |
| `hole` | simple, counterbore, countersink, drill point; at points or sketch points on a face |
| `fillet`, `chamfer` | straight edges between two planar faces (convex or concave), not meeting another blended edge |
| `shell` | an extrusion or revolution hollowed by an exact profile offset, open ends |
| `pattern`, `mirror`, `transform` | linear and circular patterns, mirrors, rigid motions |
| `boolean` | union, cut or intersection of two bodies of the part |
| `box`, `cylinder`, `sphere`, `cone`, `torus` | primitives in a frame |
| `solid` | a B-rep given as data (what STEP and IFC imports store) |

`PartHandle` adds features (`part.extrude`, `hole`, `fillet`, ..., `add_feature(dict)`),
changes them (`part.set(id, **changes)`), and reports (`part.faces()`, `part.measure()`).
Features that cannot be built are refused with the condition that failed
(`FILLET_CORNER_UNSUPPORTED`, `HOLE_OFF_FACE`, `SHELL_TOO_THICK`, ...); nothing is
approximated silently.

**Persistent names.** Faces are named by what made them: `"<feature>:<role>"` --
`base:end`, `base:side/l3` (swept by sketch line `l3`), `holes.0:rev/wall.0`. A face split
by a later feature gets `#0`, `#1`, ... (ordered by position); edges are named by their two
faces. References resolve by name, with `"near": [x, y, z]` to pick one of several pieces;
`REFERENCE_LOST` and `REFERENCE_AMBIGUOUS` list the candidates. Face and triangle indices
are never references: they change on every regeneration.

## The B-rep kernel

* **Geometry** is exact: planes, cylinders, cones, spheres, tori, linear extrusions and
  revolutions of curves, NURBS surfaces; lines, circles, ellipses and NURBS curves for
  edges (with a measured tolerance for numerically traced intersection curves, reported
  in the body's `notes`).
* **Topology**: vertices, edges, faces with loops of oriented edges, shells; `validate()`
  checks closed loops, two opposite uses of every edge of a solid, vertices on their
  curves, curves on their surfaces (within tolerance) and the Euler characteristic.
* **Tessellation** for display and analysis is derived, never stored as the model:
  edges are sampled first (so neighbouring faces share their boundary points and meshes
  are watertight), planar faces are triangulated with ear clipping, curved faces are
  meshed in their parameter space and refined until every triangle is within the chord
  tolerance (measured: `max_deviation`).
* **Mass properties** (`mass_properties(body)`: volume, area, centroid, inertia,
  principal axes) are integrated over the exact surfaces by boundary quadrature (Green's
  theorem in each face's parameter domain), not from a mesh; each result carries an
  error estimate.
* **Booleans.** Exact surface-surface booleans are a research project of their own;
  Aether3D does not claim them. It uses [Manifold](https://github.com/elalish/manifold)
  (Apache-2.0, optional dependency `manifold3d`) for robust topology on float64
  tessellations, then **reconstructs an exact B-rep** from the result: every triangle
  remembers its source face, so result faces lie on the operands' exact surfaces; edges
  are recovered analytically where the surfaces allow it (plane-plane lines, plane-quadric
  conics, coaxial circles) and traced to NURBS otherwise, with the deviation reported.
  Without `manifold3d`, features that need booleans are refused (`BOOLEAN_UNAVAILABLE`).

## Assemblies

```python
doc.add(
    "assembly",
    "Hinge",
    {
        "components": [
            {"id": "plate", "source": "Plate", "fixed": True},
            {"id": "pin", "source": "Pin", "placement": {"origin": [20, 20, 30]}},
        ],
        "mates": [
            {
                "id": "c1",
                "type": "concentric",
                "a": {"component": "plate", "face": "holes.0:rev/wall.0"},
                "b": {"component": "pin", "face": "shaft:side/wall.0"},
            },
            {
                "id": "f1",
                "type": "coincident",
                "a": {"component": "plate", "face": "base:end"},
                "b": {"component": "pin", "face": "head:start"},
            },
        ],
    },
)
```

Mates: `coincident` (planes, with `offset`, `flip`), `concentric` (axes), `parallel`,
`perpendicular`, `distance`, `angle`, `fixed`. Components are solved like a sketch over
rigid motions; `info` reports degrees of freedom per component, redundant and conflicting
mates. `clashes(doc, "Hinge")` intersects the placed bodies exactly: `interference` with
its volume, or contact. Assemblies nest; instances share their part's mesh. This is
placement, not dynamics: no masses, forces, joint limits or motion.

## Measurements

`doc.measure(a)` describes a face (surface, area, normal or axis and radii), an edge
(curve, length, radius), a sketch curve or a point; `doc.measure(a, b)` relates two:
exact `angle_deg`, `parallel_distance`, `axis_distance`, and the minimum `distance`
between the trimmed entities (sampled on the exact geometry, refined by projections onto
it, with its `bound`). References: `{"part": ..., "face" | "edge" | "vertex": ...}`,
`{"sketch": ..., "entity" | "point": ...}`, `{"point": [x, y, z]}`.

## Drawings

A `drawing` item is a view of other items, regenerated when they change:

```python
doc.drawing(
    "Section A-A",
    "section",
    ["Bracket"],
    plane={"origin": [0, 15, 0], "normal": [0, -1, 0]},
    sheet="A4",
)
d = doc.result("Section A-A")
d.to_svg()
d.to_dxf("mm")
d.to_png(150)
```

Views: `top`, `bottom`, `front`, `back`, `left`, `right`, `iso`, `section` (`plane`, the
viewer on the normal's side), and `plan` for buildings (a cut at `cut_height` above a
level). **Edges** are the exact curves sampled within the chord tolerance; silhouettes of
curved surfaces come from their tessellation; **hidden lines** are found by casting rays
toward the viewer through the tessellated solids (Aether3D's watertight ray tracer), the
split between a visible and a hidden part placed within 1/1000 of the view's extent.
**Sections** cut each solid by an exact boolean with a half-space: the hatched area is
the exact cut face. Plans add door swings, room names, numbers and areas.
**Dimensions**: automatic overall dimensions, or `dimensions: [{from, to, offset}]`.
**Sheets** A4 to A1, scales from 10:1 to 1:1000 (the largest standard scale that fits by
default), a title strip. Output: SVG (paper millimetres), DXF R12 (real size; one layer
per style, hidden lines dashed), PNG (Aether3D's own anti-aliased rasterizer and stroke
font -- no font files or image libraries). Drawings are geometric: no drafting-standard
conventions (thread symbols, broken views, GD&T).

## Exchange: STEP, DXF, IFC

| format | export | import |
| --- | --- | --- |
| STEP (ISO 10303-21) | parts' exact solids, AP214, units mm, m or in (`doc.export_step`) | manifold solid B-reps (with voids) of AP203/214/242 files: planes, cylinders, cones, spheres, tori, extrusions and revolutions of B-spline curves, B-spline surfaces; lines, circles, ellipses and B-spline curves (`doc.import_step` -> parts with a `solid` feature) |
| DXF | sketches (`doc.export_sketch_dxf`) and drawings, R12 | 2D `LINE`, `ARC`, `CIRCLE`, `LWPOLYLINE`, `POLYLINE` (with bulges; mirrored object coordinate systems) -> a sketch with shared end points (`doc.import_dxf`) |
| IFC4 | buildings ([BUILDINGS.md](BUILDINGS.md#ifc)) | the products of the same subset |
| SVG, PNG | drawings | -- |
| glTF | the shown meshes, unchanged ([GLTF.md](GLTF.md)) | -- |

**Imports are untrusted.** Strict tokenizers with size and entity limits (STEP: 200 MB,
5,000,000 entities; DXF: 64 MB; IFC: 200 MB), reference and nesting checks, finite
numbers only; nothing in a file is ever evaluated. What a reader does not support --
STEP assemblies, extrusions or revolutions of analytic curves, DXF blocks and text, IFC geometry other than extrusions and faceted
B-reps -- is refused (`UNSUPPORTED_STEP`, ...) or skipped and reported
(`import_report["skipped"]`), never guessed.

**Validated independently.** The tests read STEP files written by Open CASCADE and DXF
files written by ezdxf (fixtures in `tests/data/`), and, when
`AETHER3D_VALIDATOR_PYTHON` names a Python with `cadquery-ocp`, `ifcopenshell` and
`ezdxf`, check Aether3D's STEP with OCCT (valid shapes, volumes), its IFC with
IfcOpenShell's schema and rule validation (no issues), and its DXF with ezdxf's audit (no
errors). These tools are validators only: Aether3D never imports them.

## CAD scripts

`aether3d.cad-script` documents edit the document as data, as one undo step:

```json
{"format": "aether3d.cad-script", "version": 1, "label": "Wider plate",
 "operations": [
  {"op": "parameter", "name": "width", "value": "120 mm"},
  {"op": "update", "item": "Plate", "list": "features", "id": "holes",
   "changes": {"points": [[30, 20, 10], [90, 20, 10]]}},
  {"op": "expect", "measure": {"a": {"part": "Plate", "face": "holes.0:rev/wall.0"},
                               "b": {"part": "Plate", "face": "holes.1:rev/wall.0"}},
   "checks": {"axis_distance": {"value": 60, "tolerance": 1e-9}}}]}
```

Operations: `units`, `parameter`, `add`, `replace`, `merge`, `append` / `update` /
`delete` (elements of a definition's lists by id), `rename`, `remove`, `show`,
`convert_to_mesh`, and the read-only `info`, `measure`, `snap`, `schedule` and `expect` (a
failed check aborts the script and rolls everything back). Every length in a script is
in the document's unit or a quantity -- sketch geometry included. Limits:
`max_seconds`, `max_items`. Errors carry the step: `operations[3]: part[Plate].features[1](holes)`.
Run with `aether3d.cad.script.run_cad_script(scene, script)`, `aether3d cad run`, MCP
`run_script`, or inside a scene plan. The schema is `aether3d cad schema` (MCP resource
`aether3d://schema/cad-script`); the agent guide is `aether3d://cad/guide`.

## Command line

```
aether3d cad run PROJECT SCRIPT.json      # a CAD script, atomically; a failed expect exits 3
aether3d cad info PROJECT [ITEM]          # items, or one item's report
aether3d cad export PROJECT OUT [--item NAME ...] [--units mm] [--dpi 150]
                                          # by suffix: .step .ifc .dxf .svg .png
aether3d cad import PROJECT FILE [--name NAME] [--units mm]   # .step .ifc .dxf
aether3d cad schema                       # the CAD script JSON Schema
```

## For agents (MCP)

`run_script` runs CAD scripts; `inspect_cad` reports the document or one item (sketch
status and DOF, part validity, mass properties and named faces, quantities), optionally
with a drawing of it as an image; `import_cad` reads STEP, IFC or DXF from the workspace;
`export_scene` writes STEP, IFC, DXF, SVG or PNG by suffix. `capabilities {"topic":
"cad"}` lists operations, kinds, features, constraints, mates, views and formats. The
feedback checks `cad_valid` and `cad_dimension` ([FEEDBACK.md](FEEDBACK.md)) make CAD
facts part of `diagnose_scene`. Images are delivered to the client; whether a model sees
them is up to the client and the model -- the reports' numbers are the authority.

## What is not supported

* Exact surface-surface booleans (booleans are mesh-robust with exact reconstruction, see
  above); blends other than constant-radius fillets and chamfers of straight edges
  between planes; variable fillets, face fillets, corner blends; drafts other than on
  extrusions; offset surfaces in general (shells are profile offsets).
* NURBS surface modeling tools (lofts are ruled; sweeps follow lines and arcs).
* 3D sketches; sketch trim/extend/offset tools (the geometry routines exist internally);
  sketch splines are not constrained beyond their control points.
* Dynamic simulation of assemblies; tolerance analysis; GD&T; drafting standards.
* STEP assemblies, colours and PMI; IGES; DWG; DXF 3D entities, blocks, text, hatches on
  import; IFC2X3 and IFC geometry other than extrusions and faceted B-reps.
