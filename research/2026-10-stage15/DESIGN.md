# Stage 15 design record: precision modeling, CAD and architecture

Written on 17 October 2026 against `main` at 5e1640e (v0.14.0, Stages 1-14), before any
Stage 15 implementation code. It records what was inspected, the research behind each
decision, the implementation language chosen for every subsystem, and the test and
benchmark plans. The user guides describe what was finally built
([docs/CAD.md](../../docs/CAD.md), [docs/ARCHITECTURE_MODELING.md](../../docs/ARCHITECTURE_MODELING.md));
where they differ from this file, this file is the plan as decided and the guides are the
truth about the code. Section 10, written after the implementation, records the outcome.

## 1. Starting point (inspected)

| Area | Where | What exists | Gap for Stage 15 |
| --- | --- | --- | --- |
| Meshes | `mesh.py` | immutable float32 triangle mesh, per-vertex normals/UVs, uint16 face groups | approximate by nature; no exact geometry anywhere |
| Editing | `editing/` | BMesh-like polygon editing, journaled, scripts | polygon modeling; no dimensions, no design intent |
| Procedural | `procedural/` | nodes = source + modifier stack, SHA-256 content-addressed cache, transactions that evaluate *every* node after each change, safe `Expr` AST whitelist (unitless) | no units, no constraints, no topology names; whole-system re-check per change (fine for tens of nodes, not for thousands of elements) |
| Booleans | `procedural/modifiers.py` | Manifold (`manifold3d`, Apache-2.0, optional) with per-polygon `face_id` propagation | mesh booleans only; no analytic surfaces |
| Predicates | `spatial/predicates.py` | exact `orient2d`/`orient3d` (Shewchuk filters + `Fraction` fallback) | no incircle, nothing for curves |
| Spatial | `spatial/` | exact triangle queries, overlaps, BVHs, ray core with native traversal | clash detection exists (triangle-exact), but knows no solids or tolerances |
| History | `history.py` | aspects captured by reference, change journal for incremental caches, atomic ops, history files | no aspect for CAD definitions |
| Rendering | `rendering/` | CPU/GPU rasterizers, CPU/GPU path tracers, orthographic cameras, camera-relative float32 on the GPU | no line drawings, no text beyond a 5x7 digit bitmap |
| Interchange | `gltf/` | own validated glTF reader/writer | no CAD/BIM/2D formats |
| AI | `plans/`, `mcp/`, `feedback/`, CLI | scene plans, 23 MCP tools, `run_script` with 5 formats, expectations with exact/heuristic bases | no CAD operations |
| Native | `native/` (Rust, PyO3, abi3, ABI 2) | traversal, hierarchies, PNG filters, LDL^T, packing; every kernel bitwise equal to a NumPy twin | |
| Format | `serialization.py` | project format 7 with strict readers and migrations | nowhere to keep CAD/BIM definitions |

Conventions that bind this stage: +Z up, right-handed, metres, float64 transforms; validate
before mutating; every change through the history; immutable shared values; never call an
approximation exact; untrusted input is data; NumPy is the reference for every native kernel.

## 2. Research (sources read for this stage)

**CAD kernels considered for adoption.**
- *Open CASCADE Technology* (OCCT) through the OCP bindings: `cadquery-ocp-novtk` 8.0.1.1
  ships wheels for CPython 3.11-3.14 on Linux, macOS and Windows (66 MB each, checked with
  `pip download` on 17 Oct 2026); the bindings are Apache-2.0, OCCT itself LGPL-2.1 with an
  exception. Mature exact B-rep booleans, fillets, STEP/IGES. Against making it the kernel:
  every current dependency is permissive and optional, and Aether3D's core must work with
  `numpy` alone; the C++ API has no ABI stability (OCP pins one OCCT release; 7.7 -> 7.8 ->
  8.0 changed APIs); 66 MB native code per platform; a second, C++-owned object model beside
  the history's immutable values; its STEP reader would parse untrusted files in C++. It is
  used here as what it is best at for us: an **independent validator** of the STEP files
  Aether3D writes (reads them, checks the shape with `BRepCheck_Analyzer`, compares volumes
  and areas) -- in tests that run only where it is installed, like the Khronos glTF
  validator. It is never imported by the package.
- *truck* (Rust, Apache-2.0/MIT, NURBS B-rep): active (last push May 2026) but its solid
  booleans are young; *Fornjot* (Rust, 0BSD): early, no curved booleans. *CGAL*: GPL/LGPL,
  exact Nef polyhedra (planar only), heavy C++ templates. *Eigen* (MPL-2.0) and *faer* (MIT)
  were considered for sparse algebra (see the solver).
- *IfcOpenShell* 0.9.0 (LGPL-3.0, wheels `py310`-`py315` for all three systems): used as an
  **independent IFC validator** (schema and where-rules via `ifcopenshell.validate`,
  geometry via its OCCT-based iterator), never imported by the package.
- *ezdxf* 1.4.5 (MIT): an **independent DXF reader and auditor** in tests.

**Boundary representations.** Mantyla, *An Introduction to Solid Modeling* (Euler
operators, validity: closed oriented 2-manifold shells, Euler-Poincare); Stroud, *Boundary
Representation Modelling Techniques*; ISO 10303-42 (geometric and topological
representation) and the AP242 module *advanced boundary representation*
(ISO/TS 10303-1514: every face of a `manifold_solid_brep` is an `advanced_face`; face
geometry restricted to elementary, swept or B-spline surfaces; edges are `edge_curve`s with
explicit curves; loops are `edge_loop`s; voids are oriented closed shells with orientation
FALSE). An OCCT-written reference file was inspected for the product structure
(`PRODUCT` -> `PRODUCT_DEFINITION_FORMATION` -> `PRODUCT_DEFINITION` ->
`PRODUCT_DEFINITION_SHAPE` <- `SHAPE_DEFINITION_REPRESENTATION` ->
`ADVANCED_BREP_SHAPE_REPRESENTATION` with a `GEOMETRIC_REPRESENTATION_CONTEXT` carrying
`LENGTH_UNIT`/`PLANE_ANGLE_UNIT`/`SOLID_ANGLE_UNIT` and an
`UNCERTAINTY_MEASURE_WITH_UNIT`). Tolerant modeling (OCCT `Precision::Confusion` = 1e-7
model units; per-edge and per-vertex tolerances) is how industrial kernels store
intersection curves that have no closed form: Aether3D adopts the same honesty -- an edge
is either analytic or a B-spline with a recorded deviation.

**Manifold.** `manifold3d` 3.4.1 (the CI minimum) and 3.5.4 both provide `Mesh64`
(float64 vertex properties) and propagate per-triangle `face_id` and per-run
`run_original_id` through booleans (docs: "input faceIDs will be maintained to the
outputs"; triangles sharing an id must be coplanar -- so Aether3D passes one id per input
triangle and maps it back to the analytic face itself). Results are always manifold
(symbolic perturbation). Checked by hand in the dev venv: `Mesh64` round-trips 1e6-offset
coordinates to the last digit.

**Curves and surfaces.** Piegl & Tiller, *The NURBS Book* (2nd ed.): A2.1 `FindSpan`,
A2.2/A2.3 basis functions and derivatives, A3.2/A4.2 (rational) curve derivatives, A3.6/A4.4
surface derivatives, A5.1 knot insertion, A7.1 conics as rational quadratics (circle arcs
exactly with weights cos(theta/2)), A10.x skinning. Analytic quadrics: plane, cylinder,
cone, sphere, torus with closed-form point inversion; plane-quadric intersections are
conics (circle/ellipse/lines; parabola/hyperbola arcs are rational quadratic Beziers),
coaxial quadrics meet in circles; general quadric-quadric and anything with a torus or
NURBS is a quartic or worse and is traced numerically (Patrikalakis & Maekawa, *Shape
Interrogation for CAD and CAM*, ch. 5: marching with Newton refinement onto both surfaces).

**Robust predicates and tolerances.** Shewchuk (1997) adaptive predicates (already used);
Yap, "Towards exact geometric computation"; Hoffmann, "The problems of accuracy and
robustness in geometric computation" (1989). Policy adopted: exact predicates for
combinatorial decisions on float inputs, analytic geometry in float64, and tolerances
derived from the model's size and machine epsilon in one module, never ad-hoc epsilons.

**Geometric constraint solving.** Survey: Jermann, Trombettoni et al. / Hoffmann &
Joan-Arinyo; "A review on geometric constraint solving" (arXiv 2202.13795). Structural
decomposition by maximum bipartite matching and the Dulmage-Mendelsohn decomposition
(Ait-Aoudia, Jegou & Michelucci 1993; Trombettoni 1998): any maximum matching gives the
canonical split into under-, well- and over-constrained parts, and the well-constrained part
into a block-triangular sequence of strongly connected blocks; structure alone cannot
detect numerical dependence (parallel-and-perpendicular, a triangle of three fixed
lengths), so it is paired with numerical rank analysis. SolveSpace (`system.cpp`, read in
mirror): modified Newton with least-squares steps on the Jacobian, rank by Gram-Schmidt on
rows with a magnitude tolerance (1e-4 there), "find which to remove" by deleting candidate
constraints and re-checking rank, distinguishing redundant-but-consistent from
did-not-converge. FreeCAD PlaneGCS: DogLeg (default), Levenberg-Marquardt and BFGS;
diagnosis by rank-revealing QR (Eigen dense full-pivot, or SparseQR), groups of dependent
parameters for "partially redundant" constraints; reported LM false "fully constrained"
cases (tracker 3773) are a reminder to test DOF on known configurations.
Levenberg-Marquardt damping update after Nielsen (1999); minimum-norm Gauss-Newton steps
(J^+ r) keep an under-constrained sketch closest to where it was -- the property that
prevents branch flips.

**Persistent naming.** Kripac (1997) "A mechanism for persistently naming topological
entities in history-based parametric solid models"; Capoyleas, Chen & Hoffmann (1996);
Marcheix & Pierra (2002); FreeCAD 1.0's toponaming work (element maps from generating
operations; maintainers call it incomplete). Adopted: names from generators (feature id +
the sketch entity or role that made the face), carried through booleans by face
provenance, with a geometric signature to disambiguate split faces, and explicit
`REFERENCE_LOST` / `REFERENCE_AMBIGUOUS` errors instead of guessing.

**Mass properties.** Mirtich (1996) "Fast and accurate computation of polyhedral mass
properties" (divergence theorem); for curved faces the same surface integrals evaluated by
triangle quadrature in each face's parameter domain.

**Interchange.** ISO 10303-21 (clear text encoding), AP214/AP242 B-rep subset (above).
DXF reference (AutoCAD 2018): R2000 (AC1015) structure, `$INSUNITS`, LINE/ARC/CIRCLE/
LWPOLYLINE/TEXT/HATCH, layers. IFC4 ADD2 TC1 (ISO 16739-1:2018; schema id `IFC4`):
`IfcProject/IfcSite/IfcBuilding/IfcBuildingStorey` with `IfcRelAggregates`, elements
contained by `IfcRelContainedInSpatialStructure`, `IfcWall` + `IfcMaterialLayerSetUsage`,
`IfcOpeningElement` + `IfcRelVoidsElement`, `IfcDoor`/`IfcWindow` + `IfcRelFillsElement`,
`IfcSlab`, `IfcRoof`, `IfcStair`, `IfcColumn`, `IfcBeam`, `IfcSpace`, geometry as
`IfcExtrudedAreaSolid` / `IfcFacetedBrep`, `IfcPropertySet` + `IfcRelDefinesByProperties`,
`IfcElementQuantity` with the base quantity sets (`Qto_WallBaseQuantities`: NetSideArea
deducts openings, GrossVolume does not -- IFC 4.3.2 lexical pages), GlobalIds as the
22-character IFC base-64 encoding of 128-bit UUIDs.

**Architecture.** Wall location lines, joins (L, T, X; miter and butt cleanups), hosted
openings and families follow the common practice of BIM authoring tools (Revit, ArchiCAD,
FreeCAD BIM): openings are defined in the host wall's local frame so they follow its
changes. Stair proportion checks use criteria the *user* supplies (e.g. Blondel's
2R + T in a given range, riser and tread bounds); Aether3D asserts no building code.

## 3. Language and architecture decision per subsystem

Decided before writing the code. The rule from earlier stages holds: one production
implementation per subsystem; a native kernel only where a measured or well-established
ratio makes it matter, and then with its NumPy twin (the package must work without the
native module); no duplicate implementations built to compare.

| Subsystem | Workload | Decision | Why |
| --- | --- | --- | --- |
| Units, quantities, tolerances, expressions | parsing, scalar arithmetic | Python | not hot; correctness and clear errors matter |
| 2D/3D analytic geometry, NURBS | batches of evaluations, intersections | NumPy (vectorized over parameters) | evaluation is batch-shaped (tessellation, sampling); closed forms are a few flops |
| Sketch solver numerics | Jacobians of 10-10,000 equations, dense blocks after decomposition | NumPy residuals/Jacobians vectorized per constraint type + LAPACK (`numpy.linalg`) on the decomposed blocks | structural decomposition leaves small dense blocks; LAPACK is the right tool; a Rust solver would need a bitwise NumPy twin of a nonlinear solver with LAPACK calls -- a second solver |
| Sketch structural analysis | bipartite matching, SCCs | Python (iterative Hopcroft-Karp, Tarjan), cached per sketch *structure* | O(E sqrt V); changing a dimension does not change the structure, so it is not on the edit path |
| B-rep data | topology tables + analytic records | Python objects per body (few hundred entities), NumPy for vertices | bodies are small; arrays where batches occur |
| Polygon triangulation (every planar face, slabs, sections, hatches) | many small polygons with holes, some large ones | **Rust kernel in `aether3d-native`** (batched: many polygons per call) + Python twin, bitwise equal (integer output) | the most frequent kernel of the stage; Python ear clipping is O(n^2) per polygon with ~50 us overhead each; batching removes FFI cost; decisions use only `+ - *` and comparisons, so parity is exact |
| Curved-face tessellation | parameter-space refinement | NumPy per face | surface evaluation is vectorized; face counts are small |
| Booleans | robust mesh arrangement | **Manifold** (`manifold3d`, C++, Apache-2.0, optional) on float64 tessellations, then Aether3D's own B-rep reconstruction (Python/NumPy) | robust topology from a proven kernel; exact geometry recovered from provenance |
| Mass properties, sections, measurements | quadrature, plane cuts | NumPy | vectorized integrals |
| Assembly mates | 6 DOF per instance, small systems | NumPy (the sketch solver's numerics, generalized) | tiny dense systems |
| Architecture elements | thousands of small, structured solids | Python element generators with vectorized mesh builders for walls/openings; content-addressed caches; dirty propagation | per-element work is tiny; the cost to avoid is recomputing unrelated elements |
| Hidden lines | visibility of edge samples | the existing ray core (`SceneAccel`, native traversal when installed) | one ray core (AGENTS rule) |
| Drawing raster | anti-aliased lines, fills, stroke text | NumPy | per-primitive vectorized coverage |
| STEP / IFC (Part 21), DXF, SVG writers and readers | text generation/parsing | Python | I/O-bound; parsers must be strict and limited |
| GPU | -- | none new | CAD needs float64 and branching; display goes through the existing renderers |

## 4. Architecture

```text
aether3d/cad/              precision modeling (no scene imports except type hints)
  units.py                 quantities, dimensions, units, parsing, formatting
  tolerance.py             the tolerance policy (one place)
  expr.py                  unit-aware expressions (AST whitelist, like procedural/params)
  geom2d.py, geom3d.py     analytic primitives, intersections, projections, offsets
  nurbs.py                 B-spline / NURBS curves and surfaces
  sketch/                  entities, constraints, solver, analysis, profiles, snapping
  brep/                    topology, builders, validation, tessellation, properties,
                           booleans (Manifold + reconstruction), naming
  part.py, features.py     feature trees and their regeneration
  assembly.py              components, instances, mates
  drawing/                 views, hidden lines, sections, dimensions, font, SVG/DXF/PNG
  interop/                 STEP (Part 21 writer/reader), DXF
  document.py              CadDocument: items, dependencies, dirty regeneration,
                           transactions, links to scene objects, history aspect
  script.py, schema.py, guide.md
aether3d/arch/             architecture and BIM (imports cad)
  building.py, walls.py, openings.py, slabs.py, roofs.py, stairs.py, structure.py,
  spaces.py, families.py, quantities.py, ifc.py
```

**One parametric document per scene** (`scene.cad`, created lazily like
`scene.procedural`). It holds *items* -- parameters, frames, sketches, parts, assemblies,
building elements, drawings -- each with an immutable definition (canonical JSON values,
replaced copy-on-write, never mutated) and references to other items. Evaluation is pure
and content-addressed: an item's key is the SHA-256 of its definition and its inputs' keys;
outputs live in a byte-bounded LRU. Unlike the procedural system, a change re-evaluates only
the changed items and their dependents (a reverse-dependency index), in topological
order -- so a wall edit in a 5,000-element building touches its neighbours, hosted openings,
rooms and slabs, not the building.

**Display geometry is materialized into scene objects.** An item shown in the scene has
linked `SceneObject`s; after a successful regeneration the document writes each changed
item's tessellation into its objects (through the history's `DATA` aspect). Everything
downstream -- renderers, the retained render world (which then updates only those objects),
spatial queries, glTF, feedback -- sees ordinary meshes. A linked object refuses direct mesh
edits (`CAD_DRIVEN`); `convert_to_mesh` detaches it explicitly and reports what is lost.

**History.** One aspect per item (`cad.item`: definition + alive, captured by reference)
plus the document settings; a transaction is one history operation: definitions change,
dirty items regenerate, objects are updated, and any failure restores everything (the
default policy *rejects* a change whose regeneration fails, with the failing item, feature
and diagnostics in the error). Undo restores definitions and object meshes together; outputs
come back from the cache.

**Exact versus approximate (the precision contract).**

| Tier | What | Guarantee |
| --- | --- | --- |
| `exact` | dimensions, parameters, sketch solutions (to the solver tolerance), analytic curves/surfaces, B-reps built by extrude/revolve/sweep/loft/primitives, planar booleans, analytic quantities (wall areas, volumes from formulas) | float64 evaluation of the defining formulas; solver residuals reported |
| `analytic-recovered` | boolean results with curved faces | faces keep their exact surfaces; edges are analytic when the surface pair has a closed-form intersection, else a B-spline whose deviation from both surfaces is measured and stored; vertex tolerances stored |
| `quadrature` | mass properties of curved bodies | Gauss quadrature on exact surfaces over the trimmed domains, with an error estimate from two refinement levels |
| `tessellated` | display meshes, clash detection on curved faces, glTF | explicit chordal deviation and angle tolerances; never reported as exact |

## 5. Algorithms

**Sketch solver.**
1. Variables: point coordinates, circle radii, spline control points (arcs are center +
   two endpoints with an implicit equal-distance equation, as in SolveSpace, so endpoints
   are true points for coincidences). Lengths are model metres; angle equations are written
   as length-scaled sines/cosines so all residuals have length units (scaling).
2. Equations per constraint type, with analytic Jacobians, vectorized per type.
3. Structure: incidence graph -> connected components -> maximum matching (Hopcroft-Karp)
   -> Dulmage-Mendelsohn: over-determined part, under-determined part, well-determined
   part in block-triangular order (Tarjan SCC on the matched digraph). Cached by structure.
4. Numerics per block (in order; blocks of the under-determined part solved together):
   damped Gauss-Newton with minimum-norm steps (`lstsq` = SVD) and a Levenberg-Marquardt
   fallback (Nielsen damping); convergence on the scaled residual
   `max|r| <= tol_solve` (tolerance policy). Blocks whose structure is square but whose
   Jacobian is numerically singular fall back to solving their whole component.
5. Analysis at the solution: rank of the component Jacobian by SVD with tolerance
   `sigma_max * max(m, n) * eps * 1e3`; DOF = n - rank (per component and total).
   Dependent constraints by sequential orthogonalization of equation rows in constraint
   *creation order* (the newest constraint is blamed); a dependent equation with a residual
   above tolerance is *conflicting*, otherwise *redundant*; each report names the
   constraints it depends on (non-zero coefficients of its projection).
6. Stability: solving starts from the stored geometry; minimum-norm steps move it the least;
   after a solve, orientation signatures (signed angles of constrained corners, arc senses,
   polygon orientations) are compared with the previous solution, and a flip triggers a
   continuation solve (the changed dimensions interpolated in steps). Same inputs, same
   operations, same order: deterministic on a given platform/LAPACK.
7. Atomicity: solve on copies; the definition is replaced only on success.

**Profiles.** Non-construction curves are intersected pairwise (line/arc closed forms,
splines by subdivision + Newton), split at intersections, endpoints merged within the
sketch tolerance, and faces of the planar arrangement traced with half-edges sorted by
tangent angle then curvature at each vertex. Regions are classified (outer loops and holes
by signed area and nesting). Self-intersecting or open profiles are errors with locations.

**B-rep.** Topology tables (bodies -> shells -> faces -> loops -> coedges -> edges ->
vertices); faces reference a surface with a sense; edges a curve with a parameter range.
Builders construct exact B-reps for extrusions (lines -> planes, arcs/circles -> cylinders,
splines -> extrusion surfaces; draft -> planes/cones), revolutions (planes, cylinders,
cones, spheres, tori, revolution surfaces), sweeps along planar line/arc paths, ruled and
smooth lofts (NURBS by construction), and primitives. Validation: closed oriented
2-manifold shells, every edge used by two coedges of opposite orientation, loop closure,
vertices on curves and curves on surfaces within tolerance, Euler-Poincare.

**Booleans.** Each operand is tessellated finely (an internal tolerance far below display
tolerance), edges first so adjacent faces share samples; triangles carry their face; the
float64 meshes go through Manifold with one face id per triangle; output triangles are
grouped by source face, split into connected components, coplanar/co-surface neighbours
from the same surface are merged; boundary chains between two faces become edges, whose
curves are recovered analytically from the two surfaces (plane-plane line, plane-quadric
conics, coaxial circles, original operand edges) or traced numerically (Newton projection
onto both surfaces, B-spline fit, measured deviation); vertices are refined onto all
incident surfaces. Faces near tangency (surfaces closer than the boolean tolerance along a
chain without crossing) are reported (`NEAR_TANGENT`), never silently merged.

**Tessellation.** Edge discretization by chordal deviation and angle; planar faces
triangulated in 2D (native/twin ear clipping with hole bridging); curved faces in their
parameter domain: boundary in (u, v) with periodic unwrapping and seams, ear clipping, then
Delaunay flips and longest-edge refinement until every triangle meets the chordal and
angular tolerances; normals from the exact surface; UVs from surface parameters scaled to
metres; face groups per face class.

**Feature tree.** Ordered features in a part; references to sketches (items), to datums,
and to topology by persistent names. Regeneration replays features from the first dirty
one (each feature's output cached by its key). Supported features: extrude
(blind/symmetric/two-sided/up-to-through-all, taper/draft), revolve, sweep, loft, hole
(simple/counterbore/countersink, blind/through), fillet and chamfer (straight edges between
two planar faces; ends on planar faces), shell (single-sweep prismatic or revolved bodies,
by offsetting the profile exactly), linear/circular patterns and mirror of features or
bodies, transform, boolean between bodies. Anything outside a feature's supported class is
an error that says which condition failed.

**Assemblies.** Components (parts or sub-assemblies) with instances (local frames);
mates (fixed, coincident planes, concentric axes, parallel, perpendicular, distance, angle)
solved over 6 DOF per free instance (rotation by incremental axis-angle) with the sketch
solver's numerics and analysis (DOF, conflicts); clash detection with the spatial index on
display meshes (labeled tessellated) and exact classification for planar bodies.

**Architecture.** Buildings with levels, grids and wall types (layers); walls from 2D
location lines (explicit points or sketch lines), height to a level or explicit; joins
detected at endpoints (L, T, X) and resolved exactly in plan (offset-line intersections;
miter or butt by priority); openings hosted by walls in wall-local coordinates; doors and
windows from parametric families (frames, leaves, glazing with material groups) shared as
instances when their parameters match; slabs/ceilings from boundaries (explicit or the
inner/outer faces of a closed wall loop); roofs (flat, shed, gable, hip on convex
footprints); straight and L stairs validated against user criteria; columns and beams with
rectangular/circular/I profiles; spaces from closed boundaries, with net areas. All of them
regenerate through the document's dependency graph.

**BIM.** Each element has an IFC class and predefined type, a stable GlobalId derived from
its item id, property sets (common sets populated from parameters, user sets), materials and
quantities computed analytically where formulas exist (wall gross/net side area with
opening deductions, volumes, footprints, slab areas/perimeters/volumes, space net floor
area/perimeter, opening areas, stair counts). Schedules export as CSV/JSON. IFC4 export
writes real entities and relationships (spatial structure, aggregates, containment,
voids/fills, materials, property and quantity sets, extruded and faceted-B-rep bodies) and
is validated with IfcOpenShell; import reads back that subset.

**Drawings.** Orthographic views (plan with a cut plane, elevations, sections, isometric),
visible/hidden edges classified by rays against the display meshes with adaptive
bisection, exact section curves for planar solids (poche fills with hatching), dimension
annotations bound to model references (re-measured on every regeneration), a monoline
stroke font of Aether3D's own, output as SVG (exact vectors), DXF, and PNG rasterized by
Aether3D.

## 6. Robustness, safety and transactions

- Every input validated before any change (types, finiteness, ranges, counts, units);
  limits on entities, constraints, features, pattern counts (geometry explosion), spline
  degree (<= 9) and control points, assembly depth, expression size, file sizes, entity
  counts in imported files; solver iteration and wall-clock budgets with cancellation.
- Expressions: AST whitelist, no attribute access, no calls but listed math functions, no
  `eval`/`exec`.
- Imported STEP/DXF/IFC files: bounded tokenizers, no external references followed, all
  numbers finite and bounded, entity graphs checked for cycles before building.
- Files only through existing path policies (MCP `Workspace.path`; atomic writes).
- Failures leave the scene and its history unchanged and name the item, feature or
  constraint at fault with a stable error code and a JSON-pointer-like location.

## 7. Test plan

Independent expectations wherever possible:
- units: exact conversion factors (1 in = 25.4 mm, 1 ft = 0.3048 m), parse/format round
  trips including feet-inches-fractions, mixed-unit expressions, dimension errors;
- geometry: closed-form intersections (line/circle/arc tangencies, coincident and parallel
  cases), projections, offsets; NURBS evaluation vs. Bernstein/closed-form circles,
  derivatives vs. finite differences, knot insertion invariance;
- solver: analytic sketches (rectangles, triangles by sides, tangent arcs, symmetric
  shapes) with known coordinates; DOF of classic configurations (free point 2, segment 4,
  rectangle with h/v constraints 4, fully dimensioned 0); redundant vs conflicting
  detection and blame; branch stability over 100 random dimension edits (no orientation
  flip); atomic failure; determinism;
- B-rep: validity of every builder; volume/area/centroid/inertia vs closed forms (prisms,
  cylinders, cones, spheres, tori, revolutions via Pappus); tessellation deviation measured
  against exact surfaces; booleans vs. analytic volumes (plate with holes, slots, counterbore),
  recovered edge types, OCCT validation of STEP exports;
- features: regeneration after dimension edits, reference rebinding and its errors,
  unchanged-geometry caching, failure atomicity;
- assemblies: mate solutions vs hand-computed placements, DOF, conflicts, cycles;
- architecture: join geometry vs hand-computed miters, opening placement after wall moves,
  room areas vs formulas, quantity takeoffs vs independent arithmetic, stair criteria;
- interchange: STEP read by OCCT (valid shape, volume), IFC validated by IfcOpenShell
  (schema + rules, round trip), DXF audited by ezdxf, SVG parsed by `xml.etree`;
- persistence and history: save/load, migration from format 7, undo/redo identity,
  history files, random edit chains against fresh rebuilds (`test_invariants` style);
- AI surfaces: script schema fuzzing against the executor, CLI JSON, MCP through a real
  server, malicious payloads (NaN, huge counts, deep nesting, path escapes);
- performance regressions as same-machine ratios (`tests/timing.py`).

## 8. Benchmark plan

`benchmarks/bench_cad.py`: sketch solves (10-5,000 constraints; full and incremental),
feature regeneration (bracket; dimension edit), booleans with recovery (holes x N),
tessellation (curved faces at several tolerances), mass properties, assembly regeneration,
building generation (levels x walls x openings up to thousands of elements) and one-wall
edits (incremental vs full), drawing generation, STEP/IFC export, triangulation kernel
(native vs twin), memory (peak RSS per worker). Results in
`benchmarks/results/stage15-cad.json` with the commit measured.

## 9. Showcases

1. Bracket: constrained sketch, extrusion, holes, fillets, chamfers; hole diameter edit,
   regenerated part, residuals, validity, measurements; before/after renders and drawings;
   STEP export validated by OCCT.
2. Floor plan: connected walls from a dimensioned sketch, rooms, doors, windows, stairs,
   levels; plan, elevation/section and isometric.
3. Modern two-storey house: slabs, roof, glazing, materials, dimensions; showcase render.
4. Grid building with repeated instances: thousands of elements, one controlled edit with
   measured incremental regeneration.
5. MCP agent: create geometry, hit a conflicting constraint, read the diagnosis, fix it,
   verify measurements, render.

## 10. Outcome

Written after the implementation (release 0.15.0).

**Built as decided.** The language decisions of section 3 held: no subsystem was
implemented twice to compare. Python and NumPy/LAPACK carry units, expressions, geometry,
the sketch solver, the B-rep kernel, booleans' reconstruction, tessellation, mass
properties, assemblies, buildings, drawings and exchange; Manifold provides boolean
topology; hidden lines use the existing ray core (`SceneAccel`); the one native kernel is
the batched triangulation (`native/src/earcut.rs`, ABI 3), bitwise equal to its NumPy
twin and 41 times faster on 600 polygons (18.6 ms against 770 ms). The architecture of
section 4 is the one in the package (`cad/` and `arch/`; the architecture modules were
folded into `kinds.py`, `elements.py`, `walls.py`, `families.py`, `quantities.py` and
`ifc.py`).

**Verified independently.** Open CASCADE (OCP 8.0.1.1) reads Aether3D's STEP exports as
valid solids whose volumes match to 1e-9 relative or better (the bracket: 78337.602095844
mm3 against ours 78337.60209584454), and Aether3D reads files written by Open CASCADE
exactly (seam edges, full-circle edges, vertex loops, spheres, cones, a cut hemisphere);
IfcOpenShell 0.9.0's validation (schema and EXPRESS rules) finds no issue in the IFC
exports and computes the same volumes; ezdxf 1.4.5's audit finds no error in the DXF
exports, and an ezdxf-written DXF (bulged polylines, a mirrored arc) imports exactly.
These run in the tests when `AETHER3D_VALIDATOR_PYTHON` is set; none of them is a
dependency.

**Measured** (`benchmarks/bench_cad.py`, `benchmarks/results/stage15-cad.json`, tables
in docs/PERFORMANCE.md): a 4,999-constraint sketch solves with its analysis in 178 ms;
the bracket builds in 0.47 s; 100 holes in one feature in 1.1 s; a 10-level building of
2,200 walls, 1,000 doors and 1,000 rooms builds in 16.6 s (227 MiB) and moving one door
costs 134 ms. Four optimizations were found by measuring and are recorded with their
before/after figures in docs/PERFORMANCE.md (per-block rank checks, incremental piece
solving, one boolean per feature for disjoint tools, content-keyed rooms and openings).

**Deviations from this plan**, all documented in docs/CAD.md and docs/BUILDINGS.md:

- *Sketch analysis*: the dense SVD of section 5 is used up to the component sizes where it
  is cheap; structurally well-constrained components of 128 variables or more are checked
  per diagonal block of the Dulmage-Mendelsohn form (exact for full rank; a singular block
  falls back to the dense analysis). Edits re-solve only the pieces downstream of the
  change.
- *Booleans*: near-tangent configurations are not reported as `NEAR_TANGENT`; sweeps that
  meet tangentially are sewn instead of united (exact), and booleans that leave
  rounding-size slivers merge them into their neighbours. Faces of imported files whose
  loops wrap around an axis (bands and caps without seam edges, which booleans produce
  from seam faces) are handled in mass properties and tessellation.
- *Features*: extents are blind, symmetric, two-sided and through all ("up to" a face is
  not implemented); lofts are ruled.
- *Clash detection* intersects the placed B-reps exactly for every pair of overlapping
  boxes instead of a tessellated broad phase plus exact classification of planar bodies:
  exact everywhere, but slow for coincident curved surfaces (about 1.4 s per pin in its
  hole).
- *Walls*: crossing walls are refused (`WALLS_CROSS`: split them at the crossing) rather
  than joined as X joins; L-shaped stairs and ceilings as a separate kind were not built
  (straight flights; a slab with `top_offset` serves as a ceiling).
- *BIM*: property sets are the common sets Aether3D fills plus an identity set; user
  property sets are not supported yet.
- *Drawings*: visibility is sampled every 1/1000 of the view's extent (no adaptive
  bisection); section fills are exact for every solid (a boolean with a half-space, not
  only planar solids); dimensions are automatic overall dimensions or explicit points,
  not bound to model references.
- *Showcase 4* (the grid building) has 20 items with about 230 members; the
  "thousands of elements" case is measured by the benchmark (2,200 walls, 1,000 doors,
  1,000 rooms) instead.
